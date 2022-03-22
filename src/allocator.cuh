#pragma once


#include <cuda_runtime.h>

#include "cuda_helpers.cuh"

namespace cross {

/**
 * Adapted from https://howardhinnant.github.io/allocator_boilerplate.html
 *
 * As explained in the source, default implementations of commented-out parts are
 * provided by std::pointer_traits
 */
template <class T, unsigned int FLAGS = cudaHostAllocDefault>
class pinned_allocator
{
public:
    using value_type    = T;

//     using pointer       = value_type*;
//     using const_pointer = typename std::pointer_traits<pointer>::template
//                                                     rebind<value_type const>;
//     using void_pointer       = typename std::pointer_traits<pointer>::template
//                                                           rebind<void>;
//     using const_void_pointer = typename std::pointer_traits<pointer>::template
//                                                           rebind<const void>;

//     using difference_type = typename std::pointer_traits<pointer>::difference_type;
//     using size_type       = std::make_unsigned_t<difference_type>;

    template <class U> struct rebind {typedef pinned_allocator<U> other;};

    pinned_allocator() noexcept {}  // not required, unless used
    template <typename U> pinned_allocator(pinned_allocator<U> const&) noexcept {}

    value_type* allocate(std::size_t n)
    {
        value_type* p;
        CUCH(cudaHostAlloc(&p, n*sizeof(value_type), FLAGS));
        return p;
    }

    void deallocate(value_type* p, std::size_t) noexcept
    {
       CUCH(cudaFreeHost(p));
    }

//     value_type*
//     allocate(std::size_t n, const_void_pointer)
//     {
//         return allocate(n);
//     }

//     template <class U, class ...Args>
//     void
//     construct(U* p, Args&& ...args)
//     {
//         ::new(p) U(std::forward<Args>(args)...);
//     }

//     template <class U>
//     void
//     destroy(U* p) noexcept
//     {
//         p->~U();
//     }

//     std::size_t
//     max_size() const noexcept
//     {
//         return std::numeric_limits<size_type>::max();
//     }

//     allocator
//     select_on_container_copy_construction() const
//     {
//         return *this;
//     }

//     using propagate_on_container_copy_assignment = std::false_type;
//     using propagate_on_container_move_assignment = std::false_type;
//     using propagate_on_container_swap            = std::false_type;
//     using is_always_equal                        = std::is_empty<allocator>;
};

template <class T, class U>
bool
operator==(pinned_allocator<T> const&, pinned_allocator<U> const&) noexcept
{
    return true;
}

template <class T, class U>
bool
operator!=(pinned_allocator<T> const& x, pinned_allocator<U> const& y) noexcept
{
    return !(x == y);
}


}
