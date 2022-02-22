#pragma once

#include <cmath>
#include "types.cuh"
#include "helpers.cuh"

namespace cross {

struct assigned_work {
    dsize_t output_row;
    dsize_t worker_idx;
    dsize_t workers_for_row;
};

template<typename T>
__device__ __host__ T quadratic(
    T a,
    T b,
    T c,
    T x
) {
    return a*x*x + b*x + c;
}

struct triangle_distribution {

    inline static __host__ __device__ dsize_t num_workers_on_top_row(
        dsize_t max_rows_per_worker,
        dsize_t search_size_rows
    ) {
        // There will be at most 2*max_rows_per_thread items on top row
        return search_size_rows % (2*max_rows_per_worker);
    }

    inline static __host__ __device__ dsize_t num_total_worker_rows(
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows
    ) {
        return div_up(matrix_size_rows, max_rows_per_worker);
    }

    inline static __host__ __device__ dsize_t first_worker_on_row(
        int max_rows_per_worker,
        int workers_on_to_row,
        int worker_row
    ) {
        // r = max_rows_per_worker
        // t = workers_on_top_row
        // i = worker_y_index
        // x = worker triangle row
        // The equation:
        // r*x^2 - (3r - t)*x + (2r - t) = i
        return quadratic<int>(
            max_rows_per_worker,
            -(3*max_rows_per_worker - workers_on_to_row),
            2*max_rows_per_worker - workers_on_to_row,
            worker_row
        );
    }

    static __host__ dsize_t num_workers(
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows,
        dsize_t search_size_rows
    ) {
        dsize_t workers_on_top_row = num_workers_on_top_row(max_rows_per_worker, search_size_rows);
        dsize_t total_worker_rows = num_total_worker_rows(max_rows_per_worker, matrix_size_rows);

        // First worker on the row after the max row has ID of the number of workers required, as workers are indexed from 0
        // As rows are indexed from 1, we need to add 1 to total_worker_rows to get one beyond
        return first_worker_on_row(max_rows_per_worker, workers_on_top_row, total_worker_rows + 1);
    }

    /**
     * TODO: Describe the idea behind this distribution
     *
     * @param thread_y_index
     * @param max_rows_per_thread
     * @param num_total_rows
     * @return
     */
    static __device__ assigned_work distribute_rows(
        dsize_t worker_y_index,
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows,
        dsize_t search_size_rows
    )  {
        // TODO: Handle search_size not being the full left.y + right.y - 1
        //  probably one branch, where the first few threads create the triangle
        //  and rest create a rectangle

        // There will be at most 2*max_rows_per_thread items on top row
        dsize_t workers_on_top_row = num_workers_on_top_row(max_rows_per_worker, search_size_rows);
        dsize_t total_worker_rows = num_total_worker_rows(max_rows_per_worker, matrix_size_rows);

        // r = max_rows_per_worker
        // t = workers_on_top_row
        // i = worker_y_index
        // x = worker triangle row
        //
        // The following equation tells us the value of the first item on row x
        // in a triangle such as
        //             0
        //       1  2  3  4  5
        // 6  7  8  9 10 11 12 13 14
        //
        // Where the triangle grows by r items each row, has t items on top
        //
        // For x == 1, i == 0
        // For x == 2, i == 1
        // For x == 3, i == 6
        //
        // The equation:
        // r*x^2 - (3r - t)*x + (2r - t) = i
        //
        // If we then solve this for i, we get:
        //
        // floor(((3r - t) + sqrt((3r - t)^2 - 4r(2r - t - i))) / 2r)
        // Which is basically just the positive solution for the above quadratic equation
        // using the (-b + sqrt(b^2 - 4ac))/2a
        //
        // We need the floor as the result is in the interval of [row, row + 1) for values on the given row
        //
        // I got this by just going through few examples and interpolating polynomial through the first values
        // and seeing how they correlate

        dsize_t r = max_rows_per_worker;
        dsize_t t = workers_on_top_row;
        dsize_t i = worker_y_index;

        dsize_t tmp3r_t = 3*r - t;
        float numerator = tmp3r_t + sqrtf(tmp3r_t * tmp3r_t - 4 * r * (2 * r - t - i));

        // Row in the triangle of workers
        // Indexed from 1
        // So in the example above, 0 is on row 1, 1 2 3 4 5 are on row 2 etc.
        auto worker_row = static_cast<dsize_t>(floorf(numerator / (2*r)));

        // There might be excessive number of workers which are not needed for computation
        // due to the rounding to block sizes and warps etc.
        if (worker_row > total_worker_rows) {
            return assigned_work{
                // Return something out of bounds, for example search_size_rows
                search_size_rows,
                0,
                0
            };
        }

        // The lowest ID on the worker_row, which is the ID of the leftmost worker
        // In the example of the triangle above, it is 0, 1 or 6
        //dsize_t first_worker_row_worker = first_worker_on_row(max_rows_per_worker, workers_on_top_row, worker_row);
        // Reuse the precomputed value tmp3r_t
        dsize_t first_worker_row_worker = quadratic<int>(r, -tmp3r_t, 2*r - t, worker_row);
        // Offset of the current worker from the first worker on the row
        // In the example above for ID 5 it would be 4
        dsize_t in_worker_row_offset = worker_y_index - first_worker_row_worker;

        // Offset of the first thread in thread row from the first output row
        // In the example above, for anything in the row 1 2 3 4 5, this would be 2
        dsize_t worker_row_row_offset = (total_worker_rows - worker_row) * max_rows_per_worker;

        // Output row the worker should work with
        dsize_t worker_output_row = worker_row_row_offset + in_worker_row_offset;

        // This works for odd search sizes, as is the case with left.y + right.y - 1, which is the full search size
        // It should also work with even search sizes, where the search will be shifted by one towards the top left corner
        // or row and column zero
        dsize_t center_row = search_size_rows / 2;

        dsize_t dist_from_center = abs( static_cast<int>(worker_output_row) - static_cast<int>(center_row));

        dsize_t workers_for_row = div_up(matrix_size_rows - dist_from_center, max_rows_per_worker);

        // printf(
        //     "Y: %u, Worker row: %u, First on row: %u, Row offset: %u, In row offset: %u, Output row: %u, Center row: %u, Dist from center: %u, Workers for row: %u\n",
        //     worker_y_index,
        //     worker_row,
        //     first_worker_row_worker,
        //     worker_row_row_offset,
        //     in_worker_row_offset,
        //     worker_output_row,
        //     center_row,
        //     dist_from_center,
        //     workers_for_row
        // );

        return assigned_work{
            worker_output_row,
            total_worker_rows - worker_row,
            workers_for_row
        };
    }
};




// Simple rectangle distribution with twice the amount of workers where half of them
//  do no work and just immediately exit
struct rectangle_distribution {
    inline static __host__ __device__ dsize_t max_workers_per_row(
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows
    ) {
        return div_up(matrix_size_rows, max_rows_per_worker);
    }

    static __host__ dsize_t num_workers(
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows,
        dsize_t search_size_rows
    ) {
        return search_size_rows * max_workers_per_row(max_rows_per_worker, matrix_size_rows);
    }

    static __device__ assigned_work distribute_rows(
        dsize_t worker_y_index,
        dsize_t max_rows_per_worker,
        dsize_t matrix_size_rows,
        dsize_t search_size_rows
    ) {
        dsize_t center_row = search_size_rows / 2;
        dsize_t output_row = worker_y_index % search_size_rows;
        dsize_t worker_idx = worker_y_index / search_size_rows;

        dsize_t dist_from_center = abs( static_cast<int>(output_row) - static_cast<int>(center_row));
        return assigned_work{
            output_row,
            worker_idx,
            div_up(matrix_size_rows - dist_from_center, max_rows_per_worker)
        };
    }
};

struct no_distribution {
    static __host__ dsize_t num_workers(
        [[maybe_unused]] dsize_t max_rows_per_worker,
        [[maybe_unused]] dsize_t matrix_size_rows,
        dsize_t search_size_rows
    ) {
        return search_size_rows;
    }

    static __device__ assigned_work distribute_rows(
        dsize_t worker_y_index,
        [[maybe_unused]] dsize_t max_rows_per_worker,
        [[maybe_unused]] dsize_t matrix_size_rows,
        [[maybe_unused]] dsize_t search_size_rows
    ) {
        return assigned_work{
            worker_y_index,
            0,
            1
        };
    }
};

enum class distribution {
    none,
    rectangle,
    triangle
};

distribution from_string(const std::string& val);
std::string to_string(distribution dist);




}
