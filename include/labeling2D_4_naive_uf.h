// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_2D_4_NAIVE_UF_H_
#define YACCLAB_LABELING_2D_4_NAIVE_UF_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class naive_2d_4_uf : public Labeling2D<Connectivity2D::CONN_4> {
public:
    naive_2d_4_uf() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image

        LabelsSolver::Alloc(UPPER_BOUND_4_CONNECTIVITY); // Memory allocation of the labels solver
        LabelsSolver::Setup(); // Labels solver initialization

        // Rosenfeld Mask
        //   +-+
        //   |n|
        // +-+-+
        // |w|x|
        // +-+-+

        printf("IM HERE NAIVE!!!!!!!\n");

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
                if (img_row[c] == 0) {
                    continue;
                }
                else if (c > 0 && img_row[c-1] > 0) {
                    if (r > 0 && img_row_prev[c] > 0) {
                        img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c]); // x <- p + r
                    }
                    else {
                        img_labels_row[c] = img_labels_row[c - 1]; // x <- w
                    }
                }
                else if (r > 0 && img_row_prev[c] > 0) {
                    img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- w
                }
                else {
                    img_labels_row[c] = LabelsSolver::NewLabel();
                }
            }
        }

        // Second scan
        n_labels_ = LabelsSolver::Flatten();

        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = LabelsSolver::GetLabel(*img_row_start);
            }
        }

        LabelsSolver::Dealloc(); // Memory deallocation of the labels solver
    }

    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
    }

    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    {
        const int h = img_.rows;
        const int w = img_.cols;

        LabelsSolver::MemAlloc(UPPER_BOUND_4_CONNECTIVITY); // Equivalence solver

        // Data structure for memory test
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        LabelsSolver::MemSetup();

        // Rosenfeld Mask
        //   +-+
        //   |n|
        // +-+-+
        // |w|x|
        // +-+-+

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
                if (img_row[c] == 0) {
                    continue;
                }
                else if (c > 0 && img_row[c-1] > 0) {
                    if (r > 0 && img_row_prev[c] > 0) {
                        img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c]); // x <- p + r
                    }
                    else {
                        img_labels_row[c] = img_labels_row[c - 1]; // x <- w
                    }
                }
                else if (r > 0 && img_row_prev[c] > 0) {
                    img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- w
                }
                else {
                    img_labels_row[c] = LabelsSolver::NewLabel();
                }
            }
        }

        // Second scan
        n_labels_ = LabelsSolver::MemFlatten();

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                img_labels(r, c) = LabelsSolver::MemGetLabel(img_labels(r, c));
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        LabelsSolver::MemDealloc();
    }

private:
    double Alloc()
    {
        // Memory allocation of the labels solver
        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_4_CONNECTIVITY, perf_);
        // Memory allocation for the output image
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double ma_t = t - perf_.last();
        // Return total time
        return ls_t + ma_t;
    }
    void Dealloc() {
        LabelsSolver::Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan() {

        const int h = img_.rows;
        const int w = img_.cols;

        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization

        LabelsSolver::Setup();

        // Rosenfeld Mask
        //   +-+
        //   |n|
        // +-+-+
        // |w|x|
        // +-+-+

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
                if (img_row[c] == 0) {
                    continue;
                }
                else if (c > 0 && img_row[c-1] > 0) {
                    if (r > 0 && img_row_prev[c] > 0) {
                        img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c]); // x <- p + r
                    }
                    else {
                        img_labels_row[c] = img_labels_row[c - 1]; // x <- w
                    }
                }
                else if (r > 0 && img_row_prev[c] > 0) {
                    img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- w
                }
                else {
                    img_labels_row[c] = LabelsSolver::NewLabel();
                }
            }
        }
    }
    void SecondScan()
    {
        n_labels_ = LabelsSolver::Flatten();

        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = LabelsSolver::GetLabel(*img_row_start);
            }
        }
    }
};

#endif // !YACCLAB_LABELING_2D_4_NAIVE_UF_H_