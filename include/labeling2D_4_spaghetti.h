// This algorithm was extracted from OpenCV where Stefano Allegretti 
// used GRAPHGEN was used to  generate an algorithm for 4-connected.
// The algorithm was modified to work with YACCLAB by William Silversmith.

// Original version:
// https://raw.githubusercontent.com/opencv/opencv/5cbfe537982b85e16bbb7b29d209fe65a2b58471/modules/imgproc/src/connectedcomponents.cpp

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// 2011 Jason Newton <nevion@gmail.com>
// 2016, 2021 Costantino Grana <costantino.grana@unimore.it>
// 2016, 2021 Federico Bolelli <federico.bolelli@unimore.it>
// 2016 Lorenzo Baraldi <lorenzo.baraldi@unimore.it>
// 2016 Roberto Vezzani <roberto.vezzani@unimore.it>
// 2016 Michele Cancilla <cancilla.michele@gmail.com>
// 2021 Stefano Allegretti <stefano.allegretti@unimore.it>
//M*/


#ifndef YACCLAB_LABELING_2D_4_SAUF_H_
#define YACCLAB_LABELING_2D_4_SAUF_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class SPAGHETTI_4 : public Labeling2D<Connectivity2D::CONN_4> {
public:
    SPAGHETTI_4() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image

        LabelsSolver::Alloc(UPPER_BOUND_4_CONNECTIVITY); // Memory allocation of the labels solver
        LabelsSolver::Setup(); // Labels solver initialization

       
        // First scan

        // We work with the 4-conn Rosenfeld mask
        //   +-+
        //   |q|
        // +-+-+
        // |s|x|
        // +-+-+

        // A bunch of defines is used to check if the pixels are foreground
        // and to define actions to be performed
        {

#define CONDITION_Q img_row_prev[c] > 0
#define CONDITION_S img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 img_labels_row[c] = 0;
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel();
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_4 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_5 img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c]); // x <- s + q
        }

        // First row
        {
            unsigned char const * const img_row = img_.ptr<unsigned char>(0);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(0);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            int c = -1;

            goto fl_tree_0;
        fl_tree_0: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_2
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_tree_1: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_4
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_break:;
        }

        // Other rows
        for (int r = 1; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);
            
            int c = -1;

            goto cl_tree_0;
        cl_tree_0: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_3
                        goto cl_tree_1;
                }
                else {
                    ACTION_2
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_tree_1: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_5
                        goto cl_tree_1;
                }
                else {
                    ACTION_4
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_break:;
        }

        // undef conditions and actions
        {
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5

#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X
        }

        // Second scan
        n_labels_ = LabelsSolver::Flatten();
        
        const int pixels = h * w;
        unsigned * img_row = img_labels_.ptr<unsigned>(0);
        for (int i = 0; i < pixels; i++) {
            img_row[i] = LabelsSolver::GetLabel(img_row[i]);
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

        // First scan

        // We work with the 4-conn Rosenfeld mask
        //   +-+
        //   |q|
        // +-+-+
        // |s|x|
        // +-+-+

        // A bunch of defines is used to check if the pixels are foreground
        // and to define actions to be performed
        {

#define CONDITION_Q img_row_prev[c] > 0
#define CONDITION_S img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 img_labels_row[c] = 0;
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel();
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_4 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_5 img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c]); // x <- s + q
        }

        // First row
        {
            unsigned char const * const img_row = img_.ptr<unsigned char>(0);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(0);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            int c = -1;

            goto fl_tree_0;
        fl_tree_0: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_2
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_tree_1: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_4
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_break:;
        }

        // Other rows
        for (int r = 1; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);
            
            int c = -1;

            goto cl_tree_0;
        cl_tree_0: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_3
                        goto cl_tree_1;
                }
                else {
                    ACTION_2
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_tree_1: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_5
                        goto cl_tree_1;
                }
                else {
                    ACTION_4
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_break:;
        }

        // undef conditions and actions
        {
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5

#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X
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

                // First scan

        // We work with the 4-conn Rosenfeld mask
        //   +-+
        //   |q|
        // +-+-+
        // |s|x|
        // +-+-+

        // A bunch of defines is used to check if the pixels are foreground
        // and to define actions to be performed
        {

#define CONDITION_Q img_row_prev[c] > 0
#define CONDITION_S img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 img_labels_row[c] = 0;
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel();
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_4 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_5 img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c]); // x <- s + q
        }

        // First row
        {
            unsigned char const * const img_row = img_.ptr<unsigned char>(0);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(0);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            int c = -1;

            goto fl_tree_0;
        fl_tree_0: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_2
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_tree_1: if ((c += 1) >= w) goto fl_break;
            if (CONDITION_X) {
                ACTION_4
                    goto fl_tree_1;
            }
            else {
                ACTION_1
                    goto fl_tree_0;
            }
        fl_break:;
        }

        // Other rows
        for (int r = 1; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);
            
            int c = -1;

            goto cl_tree_0;
        cl_tree_0: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_3
                        goto cl_tree_1;
                }
                else {
                    ACTION_2
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_tree_1: if ((c += 1) >= w) goto cl_break;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    ACTION_5
                        goto cl_tree_1;
                }
                else {
                    ACTION_4
                        goto cl_tree_1;
                }
            }
            else {
                ACTION_1
                    goto cl_tree_0;
            }
        cl_break:;
        }

        // undef conditions and actions
        {
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5

#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X
        }
    }

    void SecondScan()
    {
        n_labels_ = LabelsSolver::Flatten();

        const int h = img_.rows;
        const int w = img_.cols;

        const int pixels = h * w;
        unsigned * img_row = img_labels_.ptr<unsigned>(0);
        for (int i = 0; i < pixels; i++) {
            img_row[i] = LabelsSolver::GetLabel(img_row[i]);
        }
    }
};

#endif // !YACCLAB_LABELING_2D_4_SAUF_H_