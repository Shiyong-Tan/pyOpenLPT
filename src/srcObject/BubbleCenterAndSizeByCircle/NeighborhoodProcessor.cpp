//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// NeighborhoodProcessor.cpp
//
// Code generation for function 'NeighborhoodProcessor'
//

// Include files
#include "NeighborhoodProcessor.h"
#include "BubbleCenterAndSizeByCircle_internal_types.h"
#include "BubbleCenterAndSizeByCircle_rtwutil.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

// Type Definitions
struct c_struct_T {
    coder::array<bool, 2U> bw;
    double pixel;
    int ind;
    int nhInds[9];
    int imnhInds[9];
};

// Function Declarations
static double rt_remd_snf(double u0, double u1);

// Function Definitions
static double rt_remd_snf(double u0, double u1)
{
    double y;
    if (std::isnan(u0) || std::isnan(u1) || std::isinf(u0)) {
        y = rtNaN;
    }
    else if (std::isinf(u1)) {
        y = u0;
    }
    else if ((u1 != 0.0) && (u1 != std::trunc(u1))) {
        double q;
        q = std::abs(u0 / u1);
        if (!(std::abs(q - std::floor(q + 0.5)) > DBL_EPSILON * q)) {
            y = 0.0 * u0;
        }
        else {
            y = std::fmod(u0, u1);
        }
    }
    else {
        y = std::fmod(u0, u1);
    }
    return y;
}

namespace coder {
    namespace images {
        namespace internal {
            namespace coder {
                void NeighborhoodProcessor::computeParameters(
                    const int imSize[2], const bool nhConn[9], int loffsets[9], int linds[9],
                    int soffsets[18], int interiorStart[2], int interiorEnd[2])
                {
                    int a[18];
                    int k;
                    int nz;
                    int pixelsPerImPage1_idx_1;
                    pixelsPerImPage1_idx_1 = imSize[0];
                    interiorStart[0] = 2;
                    interiorEnd[0] = imSize[0] - 1;
                    interiorStart[1] = 2;
                    interiorEnd[1] = imSize[1] - 1;
                    nz = nhConn[0];
                    for (k = 0; k < 8; k++) {
                        nz += nhConn[k + 1];
                    }
                    if (nz != 0) {
                        double b;
                        int subs_idx_1_tmp_tmp;
                        nz = 0;
                        for (int pind{ 0 }; pind < 9; pind++) {
                            if (nhConn[pind]) {
                                subs_idx_1_tmp_tmp = static_cast<int>(
                                    rt_remd_snf((static_cast<double>(pind) + 1.0) - 1.0, 3.0));
                                k = static_cast<int>(
                                    (static_cast<double>((pind - subs_idx_1_tmp_tmp) - 1) + 1.0) / 3.0);
                                soffsets[nz] = subs_idx_1_tmp_tmp + 1;
                                soffsets[nz + 9] = k + 1;
                                linds[nz] = (subs_idx_1_tmp_tmp + k * 3) + 1;
                                loffsets[nz] = (subs_idx_1_tmp_tmp + k * pixelsPerImPage1_idx_1) + 1;
                                nz++;
                            }
                        }
                        b = static_cast<double>(imSize[0]) + 2.0;
                        for (nz = 0; nz < 9; nz++) {
                            loffsets[nz] -= static_cast<int>(b);
                        }
                        std::copy(&soffsets[0], &soffsets[18], &a[0]);
                        for (k = 0; k < 2; k++) {
                            for (subs_idx_1_tmp_tmp = 0; subs_idx_1_tmp_tmp < 9;
                                subs_idx_1_tmp_tmp++) {
                                nz = subs_idx_1_tmp_tmp + 9 * k;
                                soffsets[nz] = a[nz] - 2;
                            }
                        }
                    }
                }

                void NeighborhoodProcessor::process2D(const ::coder::array<double, 2U>& in,
                    ::coder::array<bool, 2U>& out,
                    const ::coder::array<bool, 2U>& previous) const
                {
                    array<bool, 1U> out_;
                    c_struct_T fparamsAugmented;
                    double b_imnh_data[81];
                    double imnh_data[81];
                    int imnhSubs[18];
                    int b_imnhInds_data[9];
                    int c_imnhInds_data[9];
                    int d_imnhInds_data[9];
                    int imageNeighborLinearOffsets[9];
                    int imnhInds[9];
                    int imnhInds_data[9];
                    int pixelSub[2];
                    int b_firstInd;
                    int b_i;
                    int b_trueCount;
                    int c_i;
                    int c_trueCount;
                    int d_trueCount;
                    int i;
                    int i1;
                    int imageSize1;
                    int lb_loop;
                    int pind;
                    int trueCount;
                    int ub_loop;
                    signed char b_tmp_data[9];
                    signed char c_tmp_data[9];
                    signed char d_tmp_data[9];
                    signed char tmp_data[9];
                    signed char imnh_size[2];
                    bool isInside[9];
                    bool exitg1;
                    pixelSub[0] = this->InteriorStart[0];
                    pixelSub[1] = this->InteriorEnd[0];
                    for (i = 0; i < 9; i++) {
                        imageNeighborLinearOffsets[i] = this->ImageNeighborLinearOffsets[i];
                    }
                    imageSize1 = this->ImageSize[0];
                    lb_loop = this->InteriorStart[1];
                    ub_loop = this->InteriorEnd[1];
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
    out_, fparamsAugmented, imnh_data, imnh_size, imnhInds, pind, b_i, i1,     \
    b_firstInd, c_i, exitg1)

                    for (int secondInd = lb_loop; secondInd <= ub_loop; secondInd++) {
                        out_.set_size(out.size(0));
                        b_i = pixelSub[0];
                        i1 = pixelSub[1];
                        for (b_firstInd = b_i; b_firstInd <= i1; b_firstInd++) {
                            //  Process pixels with full neighborhood
                            pind = (secondInd - 1) * imageSize1 + b_firstInd;
                            for (c_i = 0; c_i < 9; c_i++) {
                                imnhInds[c_i] = imageNeighborLinearOffsets[c_i] + pind;
                            }
                            if (in.size(0) == 1) {
                                imnh_size[0] = 1;
                                imnh_size[1] = 9;
                                for (c_i = 0; c_i < 9; c_i++) {
                                    imnh_data[c_i] = in[imnhInds[c_i] - 1];
                                }
                            }
                            else {
                                imnh_size[0] = 9;
                                imnh_size[1] = 1;
                                for (c_i = 0; c_i < 9; c_i++) {
                                    imnh_data[c_i] = in[imnhInds[c_i] - 1];
                                }
                            }
                            fparamsAugmented.pixel = in[pind - 1];
                            out_[b_firstInd - 1] = previous[pind - 1];
                            if (previous[pind - 1]) {
                                //  Pixel has not already been set as non-max
                                pind = 0;
                                exitg1 = false;
                                while ((!exitg1) && (pind <= imnh_size[0] * imnh_size[1] - 1)) {
                                    if (imnh_data[pind] > fparamsAugmented.pixel) {
                                        //  Set pixel to zero if any neighbor is greater
                                        out_[b_firstInd - 1] = false;
                                        exitg1 = true;
                                    }
                                    else if ((imnh_data[pind] == fparamsAugmented.pixel) &&
                                        (!previous[imnhInds[pind] - 1])) {
                                        //  Set pixel to zero if any equal neighbor is already set to zero
                                        out_[b_firstInd - 1] = false;
                                        exitg1 = true;
                                    }
                                    else {
                                        pind++;
                                    }
                                }
                            }
                        }
                        pind = out_.size(0);
                        for (b_i = 0; b_i < pind; b_i++) {
                            out[b_i + out.size(0) * (secondInd - 1)] = out_[b_i];
                        }
                    }
                    if (this->ProcessBorder) {
                        double fparamsAugmented_pixel;
                        double padValue_tmp;
                        int b_pind;
                        int b_secondInd;
                        int firstInd;
                        int secondDimExtents_idx_0;
                        int secondDimExtents_idx_1;
                        int u1;
                        int y;
                        secondDimExtents_idx_1 = this->InteriorStart[1] - 1;
                        padValue_tmp = this->PadValue;
                        imageSize1 = this->ImageSize[0];
                        u1 = this->ImageSize[0];
                        if (imageSize1 < u1) {
                            u1 = imageSize1;
                        }
                        imageSize1 = this->ImageSize[1];
                        if (secondDimExtents_idx_1 >= imageSize1) {
                            secondDimExtents_idx_1 = imageSize1;
                        }
                        for (b_secondInd = 1; b_secondInd <= secondDimExtents_idx_1;
                            b_secondInd++) {
                            for (firstInd = 1; firstInd <= u1; firstInd++) {
                                //  Process pixels with partial neighborhood
                                b_pind = ((b_secondInd - 1) * this->ImageSize[0] + firstInd) - 1;
                                for (i = 0; i < 9; i++) {
                                    imageNeighborLinearOffsets[i] =
                                        (this->ImageNeighborLinearOffsets[i] + b_pind) + 1;
                                }
                                pixelSub[0] = this->ImageSize[0];
                                if (pixelSub[0] == 0) {
                                    imageSize1 = 0;
                                }
                                else {
                                    imageSize1 = b_pind - pixelSub[0] * div_s32(b_pind, pixelSub[0]);
                                }
                                lb_loop = b_pind - imageSize1;
                                if (pixelSub[0] == 0) {
                                    if (lb_loop == 0) {
                                        i = 0;
                                    }
                                    else if (lb_loop < 0) {
                                        i = MIN_int32_T;
                                    }
                                    else {
                                        i = MAX_int32_T;
                                    }
                                }
                                else if (pixelSub[0] == 1) {
                                    i = lb_loop;
                                }
                                else if (pixelSub[0] == -1) {
                                    i = -lb_loop;
                                }
                                else {
                                    if (lb_loop >= 0) {
                                        ub_loop = lb_loop;
                                    }
                                    else if (lb_loop == MIN_int32_T) {
                                        ub_loop = MAX_int32_T;
                                    }
                                    else {
                                        ub_loop = -lb_loop;
                                    }
                                    if (pixelSub[0] >= 0) {
                                        y = pixelSub[0];
                                    }
                                    else if (pixelSub[0] == MIN_int32_T) {
                                        y = MAX_int32_T;
                                    }
                                    else {
                                        y = -pixelSub[0];
                                    }
                                    i = div_s32(ub_loop, y);
                                    ub_loop -= i * y;
                                    if ((ub_loop > 0) && (ub_loop >= (y >> 1) + (y & 1))) {
                                        i++;
                                    }
                                    if ((lb_loop < 0) != (pixelSub[0] < 0)) {
                                        i = -i;
                                    }
                                }
                                pixelSub[1] = i + 1;
                                pixelSub[0] = imageSize1 + 1;
                                for (lb_loop = 0; lb_loop < 2; lb_loop++) {
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = ub_loop + 9 * lb_loop;
                                        imnhSubs[imageSize1] =
                                            this->NeighborSubscriptOffsets[imageSize1] + pixelSub[lb_loop];
                                    }
                                }
                                for (i = 0; i < 9; i++) {
                                    isInside[i] = true;
                                }
                                switch (static_cast<int>(this->Padding)) {
                                case 1:
                                    trueCount = 0;
                                    lb_loop = 0;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                        if (isInside[ub_loop]) {
                                            trueCount++;
                                            imnhInds_data[lb_loop] = imageNeighborLinearOffsets[ub_loop];
                                            lb_loop++;
                                        }
                                    }
                                    break;
                                case 2:
                                    trueCount = 9;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imnhInds_data[ub_loop] = imageNeighborLinearOffsets[ub_loop];
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                imnhInds_data[ub_loop] = 1;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                    }
                                    break;
                                case 3:
                                    trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[0] = 1;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[0] = this->ImageSize[0];
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[1] = 1;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[1] = this->ImageSize[1];
                                        }
                                        if (!isInside[ub_loop]) {
                                            imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                case 4:
                                    trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[0] << 1;
                                            pixelSub[0] = i;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[0] << 1) - i) + 1;
                                            pixelSub[0] = i;
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[1] << 1;
                                            pixelSub[1] = i;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[1] << 1) - i) + 1;
                                            pixelSub[1] = i;
                                        }
                                        if (!isInside[ub_loop]) {
                                            imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                }
                                if (in.size(0) == 1) {
                                    ub_loop = 1;
                                    y = trueCount;
                                    for (i = 0; i < trueCount; i++) {
                                        b_imnh_data[i] = in[imnhInds_data[i] - 1];
                                    }
                                }
                                else {
                                    ub_loop = trueCount;
                                    y = 1;
                                    for (i = 0; i < trueCount; i++) {
                                        b_imnh_data[i] = in[imnhInds_data[i] - 1];
                                    }
                                }
                                if (this->Padding == 2.0) {
                                    imageSize1 = 0;
                                    lb_loop = 0;
                                    for (i = 0; i < 9; i++) {
                                        if (!isInside[i]) {
                                            imageSize1++;
                                            tmp_data[lb_loop] = static_cast<signed char>(i + 1);
                                            lb_loop++;
                                        }
                                    }
                                    imageSize1--;
                                    for (i = 0; i <= imageSize1; i++) {
                                        b_imnh_data[tmp_data[i] - 1] = padValue_tmp;
                                    }
                                }
                                fparamsAugmented_pixel = in[b_pind];
                                out[b_pind] = previous[b_pind];
                                if (previous[b_pind]) {
                                    //  Pixel has not already been set as non-max
                                    imageSize1 = 0;
                                    exitg1 = false;
                                    while ((!exitg1) && (imageSize1 <= ub_loop * y - 1)) {
                                        if (b_imnh_data[imageSize1] > fparamsAugmented_pixel) {
                                            //  Set pixel to zero if any neighbor is greater
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else if ((b_imnh_data[imageSize1] == fparamsAugmented_pixel) &&
                                            (!previous[imnhInds_data[imageSize1] - 1])) {
                                            //  Set pixel to zero if any equal neighbor is already set to zero
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else {
                                            imageSize1++;
                                        }
                                    }
                                }
                            }
                        }
                        secondDimExtents_idx_0 = this->InteriorEnd[1] + 1;
                        secondDimExtents_idx_1 = this->ImageSize[1];
                        if (secondDimExtents_idx_0 < 1) {
                            secondDimExtents_idx_0 = 1;
                        }
                        imageSize1 = this->ImageSize[1];
                        if (secondDimExtents_idx_1 >= imageSize1) {
                            secondDimExtents_idx_1 = imageSize1;
                        }
                        for (b_secondInd = secondDimExtents_idx_0;
                            b_secondInd <= secondDimExtents_idx_1; b_secondInd++) {
                            for (firstInd = 1; firstInd <= u1; firstInd++) {
                                //  Process pixels with partial neighborhood
                                b_pind = ((b_secondInd - 1) * this->ImageSize[0] + firstInd) - 1;
                                for (i = 0; i < 9; i++) {
                                    imageNeighborLinearOffsets[i] =
                                        (this->ImageNeighborLinearOffsets[i] + b_pind) + 1;
                                }
                                pixelSub[0] = this->ImageSize[0];
                                if (pixelSub[0] == 0) {
                                    imageSize1 = 0;
                                }
                                else {
                                    imageSize1 = b_pind - pixelSub[0] * div_s32(b_pind, pixelSub[0]);
                                }
                                lb_loop = b_pind - imageSize1;
                                if (pixelSub[0] == 0) {
                                    if (lb_loop == 0) {
                                        i = 0;
                                    }
                                    else if (lb_loop < 0) {
                                        i = MIN_int32_T;
                                    }
                                    else {
                                        i = MAX_int32_T;
                                    }
                                }
                                else if (pixelSub[0] == 1) {
                                    i = lb_loop;
                                }
                                else if (pixelSub[0] == -1) {
                                    i = -lb_loop;
                                }
                                else {
                                    if (lb_loop >= 0) {
                                        ub_loop = lb_loop;
                                    }
                                    else if (lb_loop == MIN_int32_T) {
                                        ub_loop = MAX_int32_T;
                                    }
                                    else {
                                        ub_loop = -lb_loop;
                                    }
                                    if (pixelSub[0] >= 0) {
                                        y = pixelSub[0];
                                    }
                                    else if (pixelSub[0] == MIN_int32_T) {
                                        y = MAX_int32_T;
                                    }
                                    else {
                                        y = -pixelSub[0];
                                    }
                                    i = div_s32(ub_loop, y);
                                    ub_loop -= i * y;
                                    if ((ub_loop > 0) && (ub_loop >= (y >> 1) + (y & 1))) {
                                        i++;
                                    }
                                    if ((lb_loop < 0) != (pixelSub[0] < 0)) {
                                        i = -i;
                                    }
                                }
                                pixelSub[1] = i + 1;
                                pixelSub[0] = imageSize1 + 1;
                                for (lb_loop = 0; lb_loop < 2; lb_loop++) {
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = ub_loop + 9 * lb_loop;
                                        imnhSubs[imageSize1] =
                                            this->NeighborSubscriptOffsets[imageSize1] + pixelSub[lb_loop];
                                    }
                                }
                                for (i = 0; i < 9; i++) {
                                    isInside[i] = true;
                                }
                                switch (static_cast<int>(this->Padding)) {
                                case 1:
                                    b_trueCount = 0;
                                    lb_loop = 0;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                        if (isInside[ub_loop]) {
                                            b_trueCount++;
                                            b_imnhInds_data[lb_loop] = imageNeighborLinearOffsets[ub_loop];
                                            lb_loop++;
                                        }
                                    }
                                    break;
                                case 2:
                                    b_trueCount = 9;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        b_imnhInds_data[ub_loop] = imageNeighborLinearOffsets[ub_loop];
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                b_imnhInds_data[ub_loop] = 1;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                    }
                                    break;
                                case 3:
                                    b_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        b_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[0] = 1;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[0] = this->ImageSize[0];
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[1] = 1;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[1] = this->ImageSize[1];
                                        }
                                        if (!isInside[ub_loop]) {
                                            b_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                case 4:
                                    b_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        b_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[0] << 1;
                                            pixelSub[0] = i;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[0] << 1) - i) + 1;
                                            pixelSub[0] = i;
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[1] << 1;
                                            pixelSub[1] = i;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[1] << 1) - i) + 1;
                                            pixelSub[1] = i;
                                        }
                                        if (!isInside[ub_loop]) {
                                            b_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                }
                                if (in.size(0) == 1) {
                                    ub_loop = 1;
                                    y = b_trueCount;
                                    for (i = 0; i < b_trueCount; i++) {
                                        b_imnh_data[i] = in[b_imnhInds_data[i] - 1];
                                    }
                                }
                                else {
                                    ub_loop = b_trueCount;
                                    y = 1;
                                    for (i = 0; i < b_trueCount; i++) {
                                        b_imnh_data[i] = in[b_imnhInds_data[i] - 1];
                                    }
                                }
                                if (this->Padding == 2.0) {
                                    trueCount = 0;
                                    lb_loop = 0;
                                    for (i = 0; i < 9; i++) {
                                        if (!isInside[i]) {
                                            trueCount++;
                                            b_tmp_data[lb_loop] = static_cast<signed char>(i + 1);
                                            lb_loop++;
                                        }
                                    }
                                    imageSize1 = trueCount - 1;
                                    for (i = 0; i <= imageSize1; i++) {
                                        b_imnh_data[b_tmp_data[i] - 1] = padValue_tmp;
                                    }
                                }
                                fparamsAugmented_pixel = in[b_pind];
                                out[b_pind] = previous[b_pind];
                                if (previous[b_pind]) {
                                    //  Pixel has not already been set as non-max
                                    imageSize1 = 0;
                                    exitg1 = false;
                                    while ((!exitg1) && (imageSize1 <= ub_loop * y - 1)) {
                                        if (b_imnh_data[imageSize1] > fparamsAugmented_pixel) {
                                            //  Set pixel to zero if any neighbor is greater
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else if ((b_imnh_data[imageSize1] == fparamsAugmented_pixel) &&
                                            (!previous[b_imnhInds_data[imageSize1] - 1])) {
                                            //  Set pixel to zero if any equal neighbor is already set to zero
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else {
                                            imageSize1++;
                                        }
                                    }
                                }
                            }
                        }
                        secondDimExtents_idx_1 = this->InteriorStart[0] - 1;
                        u1 = this->ImageSize[0];
                        if (secondDimExtents_idx_1 >= u1) {
                            secondDimExtents_idx_1 = u1;
                        }
                        imageSize1 = this->ImageSize[1];
                        u1 = this->ImageSize[1];
                        if (imageSize1 < u1) {
                            u1 = imageSize1;
                        }
                        for (b_secondInd = 1; b_secondInd <= u1; b_secondInd++) {
                            for (firstInd = 1; firstInd <= secondDimExtents_idx_1; firstInd++) {
                                //  Process pixels with partial neighborhood
                                b_pind = ((b_secondInd - 1) * this->ImageSize[0] + firstInd) - 1;
                                for (i = 0; i < 9; i++) {
                                    imageNeighborLinearOffsets[i] =
                                        (this->ImageNeighborLinearOffsets[i] + b_pind) + 1;
                                }
                                pixelSub[0] = this->ImageSize[0];
                                if (pixelSub[0] == 0) {
                                    imageSize1 = 0;
                                }
                                else {
                                    imageSize1 = b_pind - pixelSub[0] * div_s32(b_pind, pixelSub[0]);
                                }
                                lb_loop = b_pind - imageSize1;
                                if (pixelSub[0] == 0) {
                                    if (lb_loop == 0) {
                                        i = 0;
                                    }
                                    else if (lb_loop < 0) {
                                        i = MIN_int32_T;
                                    }
                                    else {
                                        i = MAX_int32_T;
                                    }
                                }
                                else if (pixelSub[0] == 1) {
                                    i = lb_loop;
                                }
                                else if (pixelSub[0] == -1) {
                                    i = -lb_loop;
                                }
                                else {
                                    if (lb_loop >= 0) {
                                        ub_loop = lb_loop;
                                    }
                                    else if (lb_loop == MIN_int32_T) {
                                        ub_loop = MAX_int32_T;
                                    }
                                    else {
                                        ub_loop = -lb_loop;
                                    }
                                    if (pixelSub[0] >= 0) {
                                        y = pixelSub[0];
                                    }
                                    else if (pixelSub[0] == MIN_int32_T) {
                                        y = MAX_int32_T;
                                    }
                                    else {
                                        y = -pixelSub[0];
                                    }
                                    i = div_s32(ub_loop, y);
                                    ub_loop -= i * y;
                                    if ((ub_loop > 0) && (ub_loop >= (y >> 1) + (y & 1))) {
                                        i++;
                                    }
                                    if ((lb_loop < 0) != (pixelSub[0] < 0)) {
                                        i = -i;
                                    }
                                }
                                pixelSub[1] = i + 1;
                                pixelSub[0] = imageSize1 + 1;
                                for (lb_loop = 0; lb_loop < 2; lb_loop++) {
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = ub_loop + 9 * lb_loop;
                                        imnhSubs[imageSize1] =
                                            this->NeighborSubscriptOffsets[imageSize1] + pixelSub[lb_loop];
                                    }
                                }
                                for (i = 0; i < 9; i++) {
                                    isInside[i] = true;
                                }
                                switch (static_cast<int>(this->Padding)) {
                                case 1:
                                    c_trueCount = 0;
                                    lb_loop = 0;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                        if (isInside[ub_loop]) {
                                            c_trueCount++;
                                            c_imnhInds_data[lb_loop] = imageNeighborLinearOffsets[ub_loop];
                                            lb_loop++;
                                        }
                                    }
                                    break;
                                case 2:
                                    c_trueCount = 9;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        c_imnhInds_data[ub_loop] = imageNeighborLinearOffsets[ub_loop];
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                c_imnhInds_data[ub_loop] = 1;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                    }
                                    break;
                                case 3:
                                    c_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        c_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[0] = 1;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[0] = this->ImageSize[0];
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[1] = 1;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[1] = this->ImageSize[1];
                                        }
                                        if (!isInside[ub_loop]) {
                                            c_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                case 4:
                                    c_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        c_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[0] << 1;
                                            pixelSub[0] = i;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[0] << 1) - i) + 1;
                                            pixelSub[0] = i;
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[1] << 1;
                                            pixelSub[1] = i;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[1] << 1) - i) + 1;
                                            pixelSub[1] = i;
                                        }
                                        if (!isInside[ub_loop]) {
                                            c_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                }
                                if (in.size(0) == 1) {
                                    ub_loop = 1;
                                    y = c_trueCount;
                                    for (i = 0; i < c_trueCount; i++) {
                                        b_imnh_data[i] = in[c_imnhInds_data[i] - 1];
                                    }
                                }
                                else {
                                    ub_loop = c_trueCount;
                                    y = 1;
                                    for (i = 0; i < c_trueCount; i++) {
                                        b_imnh_data[i] = in[c_imnhInds_data[i] - 1];
                                    }
                                }
                                if (this->Padding == 2.0) {
                                    trueCount = 0;
                                    lb_loop = 0;
                                    for (i = 0; i < 9; i++) {
                                        if (!isInside[i]) {
                                            trueCount++;
                                            c_tmp_data[lb_loop] = static_cast<signed char>(i + 1);
                                            lb_loop++;
                                        }
                                    }
                                    imageSize1 = trueCount - 1;
                                    for (i = 0; i <= imageSize1; i++) {
                                        b_imnh_data[c_tmp_data[i] - 1] = padValue_tmp;
                                    }
                                }
                                fparamsAugmented_pixel = in[b_pind];
                                out[b_pind] = previous[b_pind];
                                if (previous[b_pind]) {
                                    //  Pixel has not already been set as non-max
                                    imageSize1 = 0;
                                    exitg1 = false;
                                    while ((!exitg1) && (imageSize1 <= ub_loop * y - 1)) {
                                        if (b_imnh_data[imageSize1] > fparamsAugmented_pixel) {
                                            //  Set pixel to zero if any neighbor is greater
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else if ((b_imnh_data[imageSize1] == fparamsAugmented_pixel) &&
                                            (!previous[c_imnhInds_data[imageSize1] - 1])) {
                                            //  Set pixel to zero if any equal neighbor is already set to zero
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else {
                                            imageSize1++;
                                        }
                                    }
                                }
                            }
                        }
                        secondDimExtents_idx_0 = this->InteriorEnd[0] + 1;
                        secondDimExtents_idx_1 = this->ImageSize[0];
                        if (secondDimExtents_idx_0 < 1) {
                            secondDimExtents_idx_0 = 1;
                        }
                        imageSize1 = this->ImageSize[0];
                        if (secondDimExtents_idx_1 >= imageSize1) {
                            secondDimExtents_idx_1 = imageSize1;
                        }
                        for (b_secondInd = 1; b_secondInd <= u1; b_secondInd++) {
                            for (firstInd = secondDimExtents_idx_0;
                                firstInd <= secondDimExtents_idx_1; firstInd++) {
                                //  Process pixels with partial neighborhood
                                b_pind = ((b_secondInd - 1) * this->ImageSize[0] + firstInd) - 1;
                                for (i = 0; i < 9; i++) {
                                    imageNeighborLinearOffsets[i] =
                                        (this->ImageNeighborLinearOffsets[i] + b_pind) + 1;
                                }
                                pixelSub[0] = this->ImageSize[0];
                                if (pixelSub[0] == 0) {
                                    imageSize1 = 0;
                                }
                                else {
                                    imageSize1 = b_pind - pixelSub[0] * div_s32(b_pind, pixelSub[0]);
                                }
                                lb_loop = b_pind - imageSize1;
                                if (pixelSub[0] == 0) {
                                    if (lb_loop == 0) {
                                        i = 0;
                                    }
                                    else if (lb_loop < 0) {
                                        i = MIN_int32_T;
                                    }
                                    else {
                                        i = MAX_int32_T;
                                    }
                                }
                                else if (pixelSub[0] == 1) {
                                    i = lb_loop;
                                }
                                else if (pixelSub[0] == -1) {
                                    i = -lb_loop;
                                }
                                else {
                                    if (lb_loop >= 0) {
                                        ub_loop = lb_loop;
                                    }
                                    else if (lb_loop == MIN_int32_T) {
                                        ub_loop = MAX_int32_T;
                                    }
                                    else {
                                        ub_loop = -lb_loop;
                                    }
                                    if (pixelSub[0] >= 0) {
                                        y = pixelSub[0];
                                    }
                                    else if (pixelSub[0] == MIN_int32_T) {
                                        y = MAX_int32_T;
                                    }
                                    else {
                                        y = -pixelSub[0];
                                    }
                                    i = div_s32(ub_loop, y);
                                    ub_loop -= i * y;
                                    if ((ub_loop > 0) && (ub_loop >= (y >> 1) + (y & 1))) {
                                        i++;
                                    }
                                    if ((lb_loop < 0) != (pixelSub[0] < 0)) {
                                        i = -i;
                                    }
                                }
                                pixelSub[1] = i + 1;
                                pixelSub[0] = imageSize1 + 1;
                                for (lb_loop = 0; lb_loop < 2; lb_loop++) {
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = ub_loop + 9 * lb_loop;
                                        imnhSubs[imageSize1] =
                                            this->NeighborSubscriptOffsets[imageSize1] + pixelSub[lb_loop];
                                    }
                                }
                                for (i = 0; i < 9; i++) {
                                    isInside[i] = true;
                                }
                                switch (static_cast<int>(this->Padding)) {
                                case 1:
                                    d_trueCount = 0;
                                    lb_loop = 0;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                        if (isInside[ub_loop]) {
                                            d_trueCount++;
                                            d_imnhInds_data[lb_loop] = imageNeighborLinearOffsets[ub_loop];
                                            lb_loop++;
                                        }
                                    }
                                    break;
                                case 2:
                                    d_trueCount = 9;
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        d_imnhInds_data[ub_loop] = imageNeighborLinearOffsets[ub_loop];
                                        imageSize1 = 0;
                                        exitg1 = false;
                                        while ((!exitg1) && (imageSize1 < 2)) {
                                            i = imnhSubs[ub_loop + 9 * imageSize1];
                                            if ((i < 1) || (i > this->ImageSize[imageSize1])) {
                                                isInside[ub_loop] = false;
                                                d_imnhInds_data[ub_loop] = 1;
                                                exitg1 = true;
                                            }
                                            else {
                                                imageSize1++;
                                            }
                                        }
                                    }
                                    break;
                                case 3:
                                    d_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        d_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[0] = 1;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[0] = this->ImageSize[0];
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i = 1;
                                            pixelSub[1] = 1;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            pixelSub[1] = this->ImageSize[1];
                                        }
                                        if (!isInside[ub_loop]) {
                                            d_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                case 4:
                                    d_trueCount = 9;
                                    for (i = 0; i < 9; i++) {
                                        d_imnhInds_data[i] = imageNeighborLinearOffsets[i];
                                    }
                                    for (ub_loop = 0; ub_loop < 9; ub_loop++) {
                                        i = imnhSubs[ub_loop];
                                        pixelSub[0] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[0] << 1;
                                            pixelSub[0] = i;
                                        }
                                        if (i > this->ImageSize[0]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[0] << 1) - i) + 1;
                                            pixelSub[0] = i;
                                        }
                                        i = imnhSubs[ub_loop + 9];
                                        pixelSub[1] = i;
                                        if (i < 1) {
                                            isInside[ub_loop] = false;
                                            i += this->ImageSize[1] << 1;
                                            pixelSub[1] = i;
                                        }
                                        if (i > this->ImageSize[1]) {
                                            isInside[ub_loop] = false;
                                            i = ((this->ImageSize[1] << 1) - i) + 1;
                                            pixelSub[1] = i;
                                        }
                                        if (!isInside[ub_loop]) {
                                            d_imnhInds_data[ub_loop] =
                                                pixelSub[0] + (pixelSub[1] - 1) * this->ImageSize[0];
                                        }
                                    }
                                    break;
                                }
                                if (in.size(0) == 1) {
                                    ub_loop = 1;
                                    y = d_trueCount;
                                    for (i = 0; i < d_trueCount; i++) {
                                        b_imnh_data[i] = in[d_imnhInds_data[i] - 1];
                                    }
                                }
                                else {
                                    ub_loop = d_trueCount;
                                    y = 1;
                                    for (i = 0; i < d_trueCount; i++) {
                                        b_imnh_data[i] = in[d_imnhInds_data[i] - 1];
                                    }
                                }
                                if (this->Padding == 2.0) {
                                    trueCount = 0;
                                    lb_loop = 0;
                                    for (i = 0; i < 9; i++) {
                                        if (!isInside[i]) {
                                            trueCount++;
                                            d_tmp_data[lb_loop] = static_cast<signed char>(i + 1);
                                            lb_loop++;
                                        }
                                    }
                                    imageSize1 = trueCount - 1;
                                    for (i = 0; i <= imageSize1; i++) {
                                        b_imnh_data[d_tmp_data[i] - 1] = padValue_tmp;
                                    }
                                }
                                fparamsAugmented_pixel = in[b_pind];
                                out[b_pind] = previous[b_pind];
                                if (previous[b_pind]) {
                                    //  Pixel has not already been set as non-max
                                    imageSize1 = 0;
                                    exitg1 = false;
                                    while ((!exitg1) && (imageSize1 <= ub_loop * y - 1)) {
                                        if (b_imnh_data[imageSize1] > fparamsAugmented_pixel) {
                                            //  Set pixel to zero if any neighbor is greater
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else if ((b_imnh_data[imageSize1] == fparamsAugmented_pixel) &&
                                            (!previous[d_imnhInds_data[imageSize1] - 1])) {
                                            //  Set pixel to zero if any equal neighbor is already set to zero
                                            out[b_pind] = false;
                                            exitg1 = true;
                                        }
                                        else {
                                            imageSize1++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            } // namespace coder
        } // namespace internal
    } // namespace images
} // namespace coder

// End of code generation (NeighborhoodProcessor.cpp)
