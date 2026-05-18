// Minimal stored-NPZ writer.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace arb {
namespace harness {

namespace detail {

inline uint32_t crc32_update(uint32_t crc, const unsigned char* data, size_t n) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1U) ? (0xedb88320U ^ (c >> 1U)) : (c >> 1U);
            }
            table[i] = c;
        }
        ready = true;
    }

    crc = crc ^ 0xffffffffU;
    for (size_t i = 0; i < n; ++i) {
        crc = table[(crc ^ data[i]) & 0xffU] ^ (crc >> 8U);
    }
    return crc ^ 0xffffffffU;
}

inline uint32_t crc32_concat(uint32_t crc, const void* data, size_t n) {
    return crc32_update(crc ^ 0xffffffffU, static_cast<const unsigned char*>(data), n) ^ 0xffffffffU;
}

inline uint32_t crc32_bytes(const void* data, size_t n) {
    return crc32_update(0U, static_cast<const unsigned char*>(data), n);
}

inline void write_u16(std::ostream& os, uint16_t v) {
    char b[2] = {
        static_cast<char>(v & 0xffU),
        static_cast<char>((v >> 8U) & 0xffU)
    };
    os.write(b, sizeof(b));
}

inline void write_u32(std::ostream& os, uint32_t v) {
    char b[4] = {
        static_cast<char>(v & 0xffU),
        static_cast<char>((v >> 8U) & 0xffU),
        static_cast<char>((v >> 16U) & 0xffU),
        static_cast<char>((v >> 24U) & 0xffU)
    };
    os.write(b, sizeof(b));
}

inline std::string make_npy_header(const std::string& descr, size_t n) {
    std::string dict = "{'descr': '" + descr +
        "', 'fortran_order': False, 'shape': (" + std::to_string(n) + ",), }";

    constexpr size_t prefix_len = 10;  // magic + version + uint16 header length
    size_t padding = 16 - ((prefix_len + dict.size() + 1) % 16);
    if (padding == 16) {
        padding = 0;
    }

    std::string header = dict;
    header.append(padding, ' ');
    header.push_back('\n');

    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("NPY header too large for v1.0");
    }
    return header;
}

inline uint32_t checked_u32(uint64_t value, const char* what) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(what) + " exceeds ZIP32 limit");
    }
    return static_cast<uint32_t>(value);
}

} // namespace detail

class StoredNpzWriter {
public:
    explicit StoredNpzWriter(const std::string& path)
        : out_(path, std::ios::binary) {
        if (!out_) {
            throw std::runtime_error("cannot open NPZ file for writing: " + path);
        }
    }

    StoredNpzWriter(const StoredNpzWriter&) = delete;
    StoredNpzWriter& operator=(const StoredNpzWriter&) = delete;

    ~StoredNpzWriter() {
        if (!closed_) {
            try {
                close();
            } catch (...) {
            }
        }
    }

    void add_f64(const std::string& name, const std::vector<double>& values) {
        add_array(name, values, "<f8");
    }

    void add_u64(const std::string& name, const std::vector<uint64_t>& values) {
        add_array(name, values, "<u8");
    }

    void add_u8(const std::string& name, const std::vector<uint8_t>& values) {
        add_array(name, values, "|u1");
    }

    void close() {
        if (closed_) {
            return;
        }

        const uint64_t central_offset_64 = tell_u64();
        for (const auto& e : entries_) {
            detail::write_u32(out_, 0x02014b50U);
            detail::write_u16(out_, 20);  // version made by
            detail::write_u16(out_, 20);  // version needed
            detail::write_u16(out_, 0);   // flags
            detail::write_u16(out_, 0);   // stored
            detail::write_u16(out_, 0);   // mtime
            detail::write_u16(out_, 0);   // mdate
            detail::write_u32(out_, e.crc);
            detail::write_u32(out_, e.size);
            detail::write_u32(out_, e.size);
            detail::write_u16(out_, static_cast<uint16_t>(e.filename.size()));
            detail::write_u16(out_, 0);  // extra length
            detail::write_u16(out_, 0);  // comment length
            detail::write_u16(out_, 0);  // disk start
            detail::write_u16(out_, 0);  // internal attrs
            detail::write_u32(out_, 0);  // external attrs
            detail::write_u32(out_, e.local_offset);
            out_.write(e.filename.data(), static_cast<std::streamsize>(e.filename.size()));
        }
        const uint64_t central_size_64 = tell_u64() - central_offset_64;

        detail::write_u32(out_, 0x06054b50U);
        detail::write_u16(out_, 0);
        detail::write_u16(out_, 0);
        detail::write_u16(out_, static_cast<uint16_t>(entries_.size()));
        detail::write_u16(out_, static_cast<uint16_t>(entries_.size()));
        detail::write_u32(out_, detail::checked_u32(central_size_64, "central directory size"));
        detail::write_u32(out_, detail::checked_u32(central_offset_64, "central directory offset"));
        detail::write_u16(out_, 0);

        closed_ = true;
        out_.close();
    }

private:
    struct Entry {
        std::string filename;
        uint32_t crc{0};
        uint32_t size{0};
        uint32_t local_offset{0};
    };

    uint64_t tell_u64() {
        const auto pos = out_.tellp();
        if (pos < 0) {
            throw std::runtime_error("failed to get NPZ output position");
        }
        return static_cast<uint64_t>(pos);
    }

    template <typename T>
    void add_array(const std::string& name, const std::vector<T>& values, const std::string& descr) {
        static_assert(std::is_trivially_copyable<T>::value, "NPZ values must be trivially copyable");
        if (closed_) {
            throw std::runtime_error("cannot add array after NPZ writer is closed");
        }
        if (entries_.size() >= std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("too many NPZ entries for ZIP32");
        }

        const std::string filename = name + ".npy";
        const std::string header = detail::make_npy_header(descr, values.size());
        const uint64_t raw_bytes_64 = static_cast<uint64_t>(values.size()) * sizeof(T);
        const uint64_t npy_bytes_64 = 6 + 2 + 2 + header.size() + raw_bytes_64;
        const uint32_t npy_bytes = detail::checked_u32(npy_bytes_64, "NPY payload");

        uint32_t crc = detail::crc32_bytes("\x93NUMPY", 6);
        const unsigned char version[2] = {1, 0};
        crc = detail::crc32_update(crc, version, 2);
        const uint16_t header_len = static_cast<uint16_t>(header.size());
        const unsigned char hlen[2] = {
            static_cast<unsigned char>(header_len & 0xffU),
            static_cast<unsigned char>((header_len >> 8U) & 0xffU)
        };
        crc = detail::crc32_update(crc, hlen, 2);
        crc = detail::crc32_update(crc, reinterpret_cast<const unsigned char*>(header.data()), header.size());
        if (!values.empty()) {
            crc = detail::crc32_update(
                crc,
                reinterpret_cast<const unsigned char*>(values.data()),
                static_cast<size_t>(raw_bytes_64)
            );
        }

        const uint32_t local_offset = detail::checked_u32(tell_u64(), "local header offset");
        detail::write_u32(out_, 0x04034b50U);
        detail::write_u16(out_, 20);
        detail::write_u16(out_, 0);
        detail::write_u16(out_, 0);
        detail::write_u16(out_, 0);
        detail::write_u16(out_, 0);
        detail::write_u32(out_, crc);
        detail::write_u32(out_, npy_bytes);
        detail::write_u32(out_, npy_bytes);
        detail::write_u16(out_, static_cast<uint16_t>(filename.size()));
        detail::write_u16(out_, 0);
        out_.write(filename.data(), static_cast<std::streamsize>(filename.size()));

        out_.write("\x93NUMPY", 6);
        out_.put(1);
        out_.put(0);
        detail::write_u16(out_, header_len);
        out_.write(header.data(), static_cast<std::streamsize>(header.size()));
        if (!values.empty()) {
            out_.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(raw_bytes_64)
            );
        }
        if (!out_) {
            throw std::runtime_error("failed while writing NPZ entry " + filename);
        }

        entries_.push_back(Entry{filename, crc, npy_bytes, local_offset});
    }

    std::ofstream out_;
    std::vector<Entry> entries_;
    bool closed_{false};
};

} // namespace harness
} // namespace arb
