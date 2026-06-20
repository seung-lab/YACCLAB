// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_SAUF_6C_H_
#define YACCLAB_LABELING_SAUF_6C_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class SAUF6C : public Labeling3D<Connectivity3D::CONN_6> {
public:
	SAUF6C() {}

	void PerformLabeling()
	{
		// img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image
		img_labels_.create(3, img_.size.p, CV_32SC1);
		memset(img_labels_.data, 0, img_labels_.total() * sizeof(int));

		LabelsSolver::Alloc(UPPER_BOUND_6_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

		// Rosenfeld Mask (z current)
		// +-+-+-+  
		// |p|q|r| 
		// +-+-+-+   
		// |s|x|
		// +-+-+

		// Rosenfeld Mask (z - 1)
		// +-+-+-+  
		// |a|b|c| 
		// +-+-+-+   
		// |d|e|f|
		// +-+-+-+
		// |g|h|i|
		// +-+-+-+

		// First scan
		for (int z = 0; z < img_.size[0]; z++) {

			unsigned char const * const img_plane = img_.data + img_.step[0] * z;   //   img_.ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(img_labels_.data) + (img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + img_.step[1] * y;
					// img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					// img_prev_plane_rows[2] = (y + 1 < img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (img_labels_.step[1] / sizeof(int)) * y;
				// labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (img_labels_.step[1] / sizeof(int));
				// labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < img_.size[2]; x++) {
					if (!img_row[x]) {
						continue;
					}
					
					if (x > 0 && img_row[x-1]) {
						if (y > 0 && img_prev_row[x]) {
							labels_row[x] = LabelsSolver::Merge(labels_row[x-1], labels_prev_row[x]);
							if (z > 0 && img_prev_plane_rows[1][x]) {
								LabelsSolver::Merge(labels_row[x], labels_prev_plane_rows[1][x]);
							}
						}
						else if (z > 0 && img_prev_plane_rows[1][x]) {
							labels_row[x] = LabelsSolver::Merge(labels_row[x-1], labels_prev_plane_rows[1][x]);
						}
						else {
							labels_row[x] = labels_row[x-1];
						}
					}
					else if (y > 0 && img_prev_row[x]) {
						if (z > 0 && img_prev_plane_rows[1][x]) {
							labels_row[x] = LabelsSolver::Merge(labels_prev_row[x], labels_prev_plane_rows[1][x]);
						}
						else {
							labels_row[x] = labels_prev_row[x];
						}
					}
					else if (z > 0 && img_prev_plane_rows[1][x]) {
						labels_row[x] = labels_prev_plane_rows[1][x];
					}
					else {
						labels_row[x] = LabelsSolver::NewLabel();
					}
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		int voxels = img_labels_.size[0] * img_labels_.size[1] * img_labels_.size[2];
		for (int i = 0; i < voxels; i++) {
			img_row[i] = LabelsSolver::GetLabel(img_row[i]);
		}

		LabelsSolver::Dealloc(); // Memory deallocation of the labels solver

	}
	
	void PerformLabelingWithSteps()	{
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

	private:
	double Alloc()
	{
		// Memory allocation of the labels solver
		double ls_t = LabelsSolver::Alloc(UPPER_BOUND_6_CONNECTIVITY, perf_);
		// Memory allocation for the output image
		perf_.start();
		img_labels_.create(3, img_.size.p, CV_32SC1);
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
		LabelsSolver::Setup(); // Labels solver initialization

		// Rosenfeld Mask (z current)
		// +-+-+-+  
		// |p|q|r| 
		// +-+-+-+   
		// |s|x|
		// +-+-+

		// Rosenfeld Mask (z - 1)
		// +-+-+-+  
		// |a|b|c| 
		// +-+-+-+   
		// |d|e|f|
		// +-+-+-+
		// |g|h|i|
		// +-+-+-+

		// First scan
		for (int z = 0; z < img_.size[0]; z++) {

			unsigned char const * const img_plane = img_.data + img_.step[0] * z;   //   img_.ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(img_labels_.data) + (img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + img_.step[1] * y;
					// img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					// img_prev_plane_rows[2] = (y + 1 < img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (img_labels_.step[1] / sizeof(int)) * y;
				// labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (img_labels_.step[1] / sizeof(int));
				// labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < img_.size[2]; x++) {
					if (!img_row[x]) {
						labels_row[x] = 0;
						continue;
					}
					
					if (x > 0 && img_row[x-1]) {
						if (y > 0 && img_prev_row[x]) {
							labels_row[x] = LabelsSolver::Merge(labels_row[x-1], labels_prev_row[x]);
							if (z > 0 && img_prev_plane_rows[1][x]) {
								LabelsSolver::Merge(labels_row[x], labels_prev_plane_rows[1][x]);
							}
						}
						else if (z > 0 && img_prev_plane_rows[1][x]) {
							labels_row[x] = LabelsSolver::Merge(labels_row[x-1], labels_prev_plane_rows[1][x]);
						}
						else {
							labels_row[x] = labels_row[x-1];
						}
					}
					else if (y > 0 && img_prev_row[x]) {
						if (z > 0 && img_prev_plane_rows[1][x]) {
							labels_row[x] = LabelsSolver::Merge(labels_prev_row[x], labels_prev_plane_rows[1][x]);
						}
						else {
							labels_row[x] = labels_prev_row[x];
						}
					}
					else if (z > 0 && img_prev_plane_rows[1][x]) {
						labels_row[x] = labels_prev_plane_rows[1][x];
					}
					else {
						labels_row[x] = LabelsSolver::NewLabel();
					}
				}
			} // Rows cycle end
		} // Planes cycle end
	}

	void SecondScan() {
		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (int z = 0; z < img_labels_.size[0]; z++) {
			for (int y = 0; y < img_labels_.size[1]; y++) {
				for (int x = 0; x < img_labels_.size[2]; x++) {
					img_row[x] = LabelsSolver::GetLabel(img_row[x]);
				}
				img_row += img_labels_.step[1] / sizeof(int);
			}
		}
	}
};

#endif // !YACCLAB_LABELING_SAUF_6C_H_