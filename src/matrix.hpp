#pragma once

#include <iterator>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string_view>

// TODO: Workaround for missing header if not C++20
#include <bit>

#include <boost/tokenizer.hpp>

#include "types.cuh"
#include "helpers.cuh"

namespace cross {

class no_padding {
public:
    static dsize2_t total_size(dsize2_t matrix_size) {
        return matrix_size;
    }

    template<typename IT>
    static void pad_row([[maybe_unused]] IT begin,[[maybe_unused]] IT end) {
        return;
    }
};

template<dsize_t MULT>
class relative_zero_padding {
public:
    static dsize2_t total_size(dsize2_t matrix_size) {
        return matrix_size * MULT;
    }

    template<typename IT>
    static void pad_row(IT begin, IT end) {
        for (auto it = begin; it != end; ++it) {
            *it = 0;
        }
    }
};

namespace impl {
    // namespace binary {

    //     template<typename T>
    //     bool try_read_value(std::istream& in, T& out) {
    //         unsigned char *out_bytes = reinterpret_cast<unsigned char*>(&out);
    //         if (!in.read(out_bytes, sizeof(T))) {
    //             if (in.gcount() != 0) {
    //                 // TODO: Invalid file format or error reading
    //                 throw std::runtime_error{"Invalid input file format"};
    //             }
    //             else {
    //                 // TODO: Most likely end of file
    //                 return false;
    //             }
    //         }

    //         if (std::endian::native == std::endian::big) {
    //             std::reverse(out_bytes, out_bytes + sizeof(T));
    //         }
    //         return true;
    //     }

    //     dsize2_t read_matrix_size(std::istream& in) {
    //         dsize_t width, height;
    //         if (!try_read_value(in, width)) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }

    //         if (!try_read_value(in, height)) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }

    //         return dsize2_t{width, height};
    //     }

    //     template <typename T>
    //     std::vector<T> read_matrix_binary(std::istream& in) {
    //         // TODO: Implement binary file loading
    //     }

    // }

    namespace csv {
        inline std::tuple<dsize2_t, dsize_t> read_header(std::istream& in) {
            std::string line;
            if (!std::getline(in, line)) {
                throw std::runtime_error{"Could not read file header"};
            }

            if (line.compare(0, 2, "# ") != 0) {
                throw std::runtime_error{"Invalid file header format"};
            }

            boost::tokenizer<boost::escaped_list_separator<char>> tok(std::string_view(line).substr(2));
            auto it = tok.begin();
            if (it == tok.end()) {
                throw std::runtime_error{"Missing matrix header"};
            }
            dsize_t width = std::stoi(*it);

            if (++it == tok.end()) {
                throw std::runtime_error("Missing matrix height");
            }
            dsize_t height = std::stoi(*it);

            dsize_t num_matrices = 1;
            if (++it != tok.end()) {
                num_matrices = std::stoi(*it);

                if (++it != tok.end()) {
                    throw std::runtime_error("Invalid matrix header format, too many tokens");
                }
            }


            return {dsize2_t{width, height}, num_matrices};
        }

        template <typename T, typename PADDING>
        void read_data(std::istream& in, dsize2_t matrix_size, dsize2_t padded_size, T* out) {
            for (dsize_t y = 0; y < matrix_size.y; ++y) {
                std::string line;
                if (!std::getline(in, line)) {
                    throw std::runtime_error{std::string{"Failed to read line "} + std::to_string(y) + " from csv file"};
                }

                boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
                auto it = tok.begin();
                dsize_t x = 0;
                for (;it != tok.end() && x < matrix_size.x; ++x, ++it) {
                    out[dsize2_t{x, y}.linear_idx(padded_size.x)] = from_string<T>(*it);
                }

                if (it != tok.end()) {
                    throw std::runtime_error{std::string{"Line "} + std::to_string(y) + " is too long, expected " + std::to_string(matrix_size.x) + "values"};
                }

                if (x != matrix_size.x) {
                    throw std::runtime_error{std::string{"Line"} + std::to_string(y) + " is too short, expected " + std::to_string(matrix_size.x) + ", got " + std::to_string(x) + " values"};
                }

                dsize_t padding_start_idx = dsize2_t{x, y}.linear_idx(padded_size.x);
                dsize_t padding_end_idx = padding_start_idx + padded_size.x - x;
                PADDING::pad_row(
                    out + padding_start_idx,
                    out + padding_end_idx
                );
            }
            // TODO: Check that we read the whole file

            for (dsize_t y = matrix_size.y; y < padded_size.y; ++y) {
                PADDING::pad_row(
                    out + dsize2_t{0, y}.linear_idx(padded_size.x),
                    out + dsize2_t{padded_size.x, y}.linear_idx(padded_size.x)
                );
            }
        }

        inline void write_header(std::ostream& out, dsize2_t size, dsize_t num_matrices = 1) {
            out << "# " << size.x << "," << size.y << "," << num_matrices << "\n";
        }

        template <typename T>
        void write_data(std::ostream& out, dsize2_t matrix_size, const T* data) {
            // TODO: Set precision
            out << std::fixed;
            for (dsize_t y = 0; y < matrix_size.y; ++y) {
                auto col_sep = "";
                for (dsize_t x = 0; x < matrix_size.x; ++x) {
                    out << col_sep << data[y * matrix_size.y + x];
                    col_sep = ",";
                }
                out << "\n";
            }
        }

    }
}


template<typename T>
class submatrix_view {
public:
    using value_type = T;
    using size_type = dsize2_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    template<typename VAL>
    class iter {
    public:
        using difference_type = ddiff_t;
        using value_type = VAL;
        using pointer = VAL*;
        using reference = VAL&;
        using iterator_category = std::random_access_iterator_tag;


        explicit iter(submatrix_view<T>& src) : src_(src), pos_(0,0) {}

        inline iter& operator+=(difference_type rhs) {
            pos_ = shift_pos(pos_, rhs, src_.size().x);
            return *this;
        }
        inline iter& operator-=(difference_type rhs) {
            pos_ = shift_pos(pos_, -rhs, src_.size().x);
            return *this;
        }
        inline value_type& operator*() const {return src_[pos_];}
        inline value_type* operator->() const {return &src_[pos_];}
        inline value_type& operator[](difference_type rhs) const {return src_[shift_pos(pos_, rhs, src_.size().x)];}

        inline iter& operator++() {pos_ = shift_pos(pos_, 1, src_.size().x); return *this;}
        inline iter& operator--() {pos_ = shift_pos(pos_, -1, src_.size().x); return *this;}
        inline iter operator++(int) const {iter tmp(*this); ++(*this); return tmp;}
        inline iter operator--(int) const {iter tmp(*this); --(*this); return tmp;}

        inline difference_type operator-(const iter& rhs) const {
            auto x_diff = (difference_type)pos_.x - (difference_type)rhs.pos_.x;
            auto y_diff = (difference_type)pos_.y - (difference_type)rhs.pos_.y;
            return y_diff * (difference_type)src_.size().x + x_diff;
        }
        inline iter operator+(difference_type rhs) const {
            iter tmp(*this);
            tmp += rhs;
            return tmp;
        }
        inline iter operator-(difference_type rhs) const {
            iter tmp(*this);
            tmp -= rhs;
            return tmp;
        }

        inline bool operator==(const iter& rhs) const {return pos_ == rhs.pos_;}
        inline bool operator!=(const iter& rhs) const {return pos_ != rhs.pos_;}
        inline bool operator>(const iter& rhs) const {
            return pos_.y > rhs.pos_.y || (pos_.y == rhs.pos_.y && pos_.x > rhs.pos_.x);
        }
        inline bool operator<(const iter& rhs) const {
            return !(*this > rhs) && !(*this == rhs);
        }
        inline bool operator>=(const iter& rhs) const {
            return !(*this < rhs);
        }
        inline bool operator<=(const iter& rhs) const {
            return !(*this > rhs);
        }
    private:
        static inline dsize2_t shift_pos(dsize2_t pos, difference_type shift, dsize_t row_size) {
            auto x = (difference_type)pos.x + shift;

            pos.y += x / (difference_type)row_size;

            auto x_shifted = x % (difference_type)row_size;
            pos.x = x_shifted >= 0 ? x_shifted : row_size + x_shifted;
            return pos;
        }

        submatrix_view<T>& src_;
        dsize2_t pos_;
    };

    using iterator = iter<value_type>;
    using const_iterator = iter<const value_type>;

    static submatrix_view<T> from_positions(size_type top_left, size_type bottom_right, dsize_t src_row_size, T* src_data) {
        return submatrix_view<T>{top_left, bottom_right - top_left, src_row_size, src_data};
    }

    static submatrix_view<T> from_position_size(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, T* src_data) {
        return submatrix_view<T>{top_left, size, src_row_size, src_data};
    }

    submatrix_view(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, pointer src_data)
        :top_left_(top_left), size_(size), src_row_size_(src_row_size), src_data_(src_data)
    { }

    dsize2_t size() const {
        return size_;
    }

    dsize_t area() const {
        return size_.area();
    }

    const_iterator begin() const {
        return iterator(*this);
    }

    iterator begin() {
        return iterator(*this);
    }

    const_iterator end() const {
        return iterator(*this) + area();
    }

    iterator end() {
        return iterator(*this) + area();
    }

    reference operator [](dsize2_t i) {
        return src_data_[(top_left_.y + i.y) * src_row_size_ + top_left_.x + i.x];
    }

    const_reference operator [](dsize2_t i) const {
        return src_data_[(top_left_.y + i.y) * src_row_size_ + top_left_.x + i.x];
    }

    submatrix_view<T> submatrix_from_positions(
        dsize2_t top_left,
        dsize2_t bottom_right
    ) const {
        return submatrix_view<T>::from_positions(
            top_left_ + top_left,
            top_left_ + bottom_right,
            src_row_size_,
            src_data_
        );
    }

    submatrix_view<T> submatrix_from_position_size(
        dsize2_t top_left,
        dsize2_t size
    ) const {
        return submatrix_view<T>::from_position_size(
            top_left_ + top_left,
            size,
            src_row_size_,
            src_data_
        );
    }
protected:
    dsize2_t top_left_;
    dsize2_t size_;
    dsize_t src_row_size_;
    pointer src_data_;
};

template<typename T>
class matrix_view: public submatrix_view<T> {
public:
    using value_type = T;
    using size_type = dsize2_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    matrix_view(dsize2_t size, pointer data)
        :submatrix_view<T>(dsize2_t(0,0), size, size.x, data)
    { }

    pointer data() {
        return this->src_data_;
    }

    const_pointer data() const {
        return this->src_data_;
    }

    const_iterator begin() const {
        return data();
    }

    iterator begin() {
        return data();
    }

    const_iterator end() const {
        return data() + this->area();
    }

    iterator end() {
        return data() + this->area();
    }
};

template<typename T, typename ALLOC = std::allocator<T>>
class data_array {
public:
    using value_type = T;
    using allocator_type = ALLOC;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<ALLOC>::pointer;
    using const_pointer = typename std::allocator_traits<ALLOC>::const_pointer;

    using iterator = pointer;
    using const_iterator = const_pointer;

    // static matrix<T, ALLOC> load_from_binary(std::istream& in) {

    //     std::vector<T> data(width * height);
    //     for (std::size_t i = 0; i < width * height; ++i) {
    //         if (!impl::try_read_value(in, data[i])) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }
    //     }

    //     return matrix{};
    // }

    data_array()
        :data_array<T, ALLOC>(dsize2_t{0,0}, 0)
    { }

    data_array(dsize2_t matrix_size, dsize_t num_matrices = 1)
        :num_matrices_(num_matrices), matrix_size_(matrix_size), data_(matrix_size.area() * num_matrices)
    { }

    template<typename PADDING>
    static std::tuple<data_array<T, ALLOC>, std::vector<dsize_t>>  load_from_csv(std::vector<std::ifstream> in) {
        std::optional<dsize2_t> matrix_size;
        std::optional<dsize2_t> padded_size;
        std::vector<dsize_t> num_matrices(in.size());

        dsize_t total_data_size = 0;
        dsize_t total_num_matrices = 0;
        for (auto&& i : in) {
            auto [in_matrix_size, in_num_matrices] = impl::csv::read_header(i);
            auto in_padded_size = PADDING::total_size(in_matrix_size);
            if (!matrix_size.has_value()) {
                matrix_size = in_matrix_size;
                padded_size = in_padded_size;
            } else if (*matrix_size != in_matrix_size || *padded_size != in_padded_size) {
                throw std::runtime_error{"Data array contains matrices of different sizes"};
            }
            total_data_size += padded_size->area();
            total_num_matrices += in_num_matrices;
            num_matrices.push_back(in_num_matrices);
        }


        std::vector<T, ALLOC> data(total_data_size);
        auto mat_data = data.data();
        for (dsize_t i = 0; i < in.size(); ++i) {
            for (dsize_t matrix = 0; matrix < num_matrices[i]; ++matrix) {
                impl::csv::read_data<T, PADDING>(in[i], *matrix_size, *padded_size, mat_data);
                mat_data += padded_size->area();
            }
        }

        return {
            data_array<T, ALLOC>{*padded_size, total_num_matrices, std::move(data)},
            std::move(num_matrices)
        };
    }

    template<typename PADDING>
    static data_array<T, ALLOC> load_from_csv(std::ifstream& in) {
        auto [matrix_size, num_matrices] = impl::csv::read_header(in);

        auto padded_matrix_size = PADDING::total_size(matrix_size);
        auto total_data_size = padded_matrix_size.area() * num_matrices;

        std::vector<T, ALLOC> data(total_data_size);
        auto mat_data = data.data();
        for (dsize_t i = 0; i < num_matrices; ++i) {
            impl::csv::read_data<T, PADDING>(in, matrix_size, padded_matrix_size, mat_data);
            mat_data += padded_matrix_size.area();
        }
        return data_array<T, ALLOC>{padded_matrix_size, num_matrices, std::move(data)};
    }

    void store_to_csv(std::vector<std::ofstream> outputs) const {
        for (dsize_t i = 0; i < outputs.size(); ++i) {
            impl::csv::write_header(outputs[i], matrix_size_);
            impl::csv::write_data(outputs[i], matrix_size_, data_.data() + matrix_size_.area() * i);
        }
    }

    void store_to_csv(std::ofstream& output) const {
        impl::csv::write_header(output, matrix_size_, num_matrices_);

        for (dsize_t i = 0; i < num_matrices_; ++i) {
            impl::csv::write_data(output, matrix_size_, data_.data() + matrix_size_.area() * i);
        }
    }

    size_type size() const {
        return num_matrices_ * matrix_size_.area();
    }

    size_type num_matrices() const {
        return num_matrices_;
    }

    dsize2_t matrix_size() const {
        return matrix_size_;
    }

    pointer data() {
        return data_.data();
    }

    const_pointer data() const {
        return data_.data();
    }

    iterator begin() {
        return data();
    }

    const_iterator begin() const {
        return data();
    }

    iterator end() {
        return data() + size();
    }

    const_iterator end() const {
        return data() + size();
    }

    matrix_view<T> view(dsize_t matrix_index = 0) {
        return matrix_view<T>{
            matrix_size_,
            data_.data() + matrix_size_.area() * matrix_index
        };
    }

    matrix_view<const T> view(dsize_t matrix_index = 0) const {
        return matrix_view<const T>{
            matrix_size_,
            data_.data() + matrix_size_.area() * matrix_index
        };
    }

private:
    dsize_t num_matrices_;
    dsize2_t matrix_size_;

    std::vector<T, ALLOC> data_;

    data_array(dsize2_t matrix_size, dsize_t num_matrices, std::vector<T, ALLOC>&& data)
        :num_matrices_(num_matrices), matrix_size_(matrix_size), data_(std::move(data))
    { }
};

}