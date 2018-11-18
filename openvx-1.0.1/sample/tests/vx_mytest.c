/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file
 * \brief The OpenVX implementation unit test code.
 * \author Erik Rainey <erik.rainey@gmail.com>
 */

#include <VX/vx.h>
#include <VX/vxu.h>

#include <VX/vx_lib_debug.h>
#include <VX/vx_lib_extras.h>
#include <VX/vx_lib_xyz.h>

#if defined(EXPERIMENTAL_USE_NODE_MEMORY)
#include <VX/vx_khr_node_memory.h>
#endif

#if defined(EXPERIMENTAL_USE_DOT)
#include <VX/vx_khr_dot.h>
#endif

#if defined(EXPERIMENTAL_USE_XML)
#include <VX/vx_khr_xml.h>
#endif

#if defined(EXPERIMENTAL_USE_TARGET)
#include <VX/vx_ext_target.h>
#endif

#if defined(EXPERIMENTAL_USE_VARIANTS)
#include <VX/vx_khr_variants.h>
#endif

#include <VX/vx_helper.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>

#define CHECK_ALL_ITEMS(array, iter, status, label) { \
    status = VX_SUCCESS; \
    for ((iter) = 0; (iter) < dimof(array); (iter)++) { \
        if ((array)[(iter)] == 0) { \
            printf("Item %u in "#array" is null!\n", (iter)); \
            assert((array)[(iter)] != 0); \
            status = VX_ERROR_NOT_SUFFICIENT; \
        } \
    } \
    if (status != VX_SUCCESS) { \
        goto label; \
    } \
}

static void vx_print_log(vx_reference ref)
{
    char message[VX_MAX_LOG_MESSAGE_LEN];
    vx_uint32 errnum = 1;
    vx_status status = VX_SUCCESS;
    do {
        status = vxGetLogEntry(ref, message);
        if (status != VX_SUCCESS)
            printf("[%05u] error=%d %s", errnum++, status, message);
    } while (status != VX_SUCCESS);
}

vx_status vx_mytest_cannyedge()
{
    vx_status status = VX_FAILURE;
    vx_context context = vxCreateContext();
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vxRegisterHelperAsLogReader(context);
        vx_uint32 i = 0, w = 640, h = 480, w2 = 300, h2 = 200, x = 0, y = 0;
        vx_uint32 range = 256, windows = 16;
        vx_image images[] = {
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 0: luma */
#if 1 
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 15: Canny */
#else	    
	    vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 1: scaled */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_S16),             /* 2: grad_x */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_S16),             /* 3: grad_y */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_S16),             /* 4: mag */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 5: phase */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 6: LUT out */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 7: AbsDiff */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 8: Threshold */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U32),             /* 9: Integral */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 10: Eroded */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 11: Dilated */
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 12: Median */
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 13: Box */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_S16),             /* 14: UpDepth */
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 15: Canny */
            vxCreateImage(context, w, h, VX_DF_IMAGE_U8),                /* 16: EqualizeHistogram */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),              /* 17: DownDepth Gy */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),                /* 18: Remapped */
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),                /* 19: Remapped 2*/
            vxCreateImage(context, w2, h2, VX_DF_IMAGE_U8),                /* 20: Diff */
#endif	    
        };
        vx_lut lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
        vx_uint8 *tmp = NULL;
        vx_int32 *histogram = NULL;
        vx_distribution dist = vxCreateDistribution(context, windows, 0, range);
        vx_float32 mean = 0.0f, stddev = 0.0f;
        vx_scalar s_mean = vxCreateScalar(context, VX_TYPE_FLOAT32, &mean);
        vx_scalar s_stddev = vxCreateScalar(context, VX_TYPE_FLOAT32, &stddev);
        vx_threshold thresh = vxCreateThreshold(context, VX_THRESHOLD_TYPE_BINARY, VX_TYPE_UINT8);
        vx_int32 lo = 140;
        vx_scalar minVal = vxCreateScalar(context, VX_TYPE_UINT8, NULL);
        vx_scalar maxVal = vxCreateScalar(context, VX_TYPE_UINT8, NULL);
        vx_array minLoc = vxCreateArray(context, VX_TYPE_COORDINATES2D, 1);
        vx_array maxLoc = vxCreateArray(context, VX_TYPE_COORDINATES2D, 1);
        vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
        vx_int32 lower = 40, upper = 250;
        vx_enum policy = VX_CONVERT_POLICY_SATURATE;
        vx_int32 shift = 7;
        vx_scalar sshift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
        vx_int32 noshift = 0;
        vx_scalar snoshift = vxCreateScalar(context, VX_TYPE_INT32, &noshift);
        vx_remap table = vxCreateRemap(context, w, h, w2, h2);
        vx_float32 dx = (vx_float32)w/w2, dy = (vx_float32)h/h2;

        CHECK_ALL_ITEMS(images, i, status, exit);

        status = vxAccessLUT(lut, (void **)&tmp, VX_WRITE_ONLY);
        if (status == VX_SUCCESS)
        {
            for (i = 0; i < range; i++)
            {
                vx_uint32 g = (vx_uint32)pow(i,1/0.93);
                tmp[i] = (vx_uint8)(g >= range ? (range - 1): g);
            }
            status = vxCommitLUT(lut, tmp);
        }

        /* create the remapping for "image scaling" using remap */
        for (y = 0; y < h2; y++)
        {
            for (x = 0; x < w2; x++)
            {
                vx_float32 nx = dx*x;
                vx_float32 ny = dy*y;
                //printf("Setting point %lu, %lu from %lf, %lf (dx,dy=%lf,%lf)\n", x, y, nx, ny, dx, dy);
                vxSetRemapPoint(table, x, y, nx, ny);
            }
        }

        status |= vxSetThresholdAttribute(thresh, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &lo, sizeof(lo));
        status |= vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
        status |= vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
        status |= vxLoadKernels(context, "openvx-debug");
        if (status == VX_SUCCESS)
        {
            vx_graph graph = vxCreateGraph(context);
            if (vxGetStatus((vx_reference)graph) == VX_SUCCESS)
            {
                vx_node nodes[] = {
                    vxFReadImageNode(graph, "bikegray_640x480.pgm", images[0]),
#if 1 
                    vxCannyEdgeDetectorNode(graph, images[0], hyst, 3, VX_NORM_L1, images[1]),
                    vxFWriteImageNode(graph, images[1], "obikecanny_640x480_P400.pgm"),
#else		    
                    vxMedian3x3Node(graph, images[0], images[12]),
                    vxBox3x3Node(graph, images[0], images[13]),
                    vxScaleImageNode(graph, images[0], images[1], VX_INTERPOLATION_TYPE_AREA),
                    vxTableLookupNode(graph, images[1], lut, images[6]),
                    vxHistogramNode(graph, images[6], dist),
                    vxSobel3x3Node(graph, images[1], images[2], images[3]),
                    vxMagnitudeNode(graph, images[2], images[3], images[4]),
                    vxPhaseNode(graph, images[2], images[3], images[5]),
                    vxAbsDiffNode(graph, images[6], images[1], images[7]),
                    vxConvertDepthNode(graph, images[7], images[14], policy, sshift),
                    vxMeanStdDevNode(graph, images[7], s_mean, s_stddev),
                    vxThresholdNode(graph, images[6], thresh, images[8]),
                    vxIntegralImageNode(graph, images[7], images[9]),
                    vxErode3x3Node(graph, images[8], images[10]),
                    vxDilate3x3Node(graph, images[8], images[11]),
                    vxMinMaxLocNode(graph, images[7], minVal, maxVal, minLoc, maxLoc, 0, 0),
                    vxCannyEdgeDetectorNode(graph, images[0], hyst, 3, VX_NORM_L1, images[15]),
                    vxEqualizeHistNode(graph, images[0], images[16]),
                    vxConvertDepthNode(graph, images[3], images[17], policy, snoshift),
                    vxRemapNode(graph, images[0], table, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, images[18]),
                    vxRemapNode(graph, images[0], table, VX_INTERPOLATION_TYPE_BILINEAR, images[19]),
                    vxAbsDiffNode(graph, images[18], images[19], images[20]),

                    vxFWriteImageNode(graph, images[0], "obikegray_640x480_P400.pgm"),
                    vxFWriteImageNode(graph, images[1], "obikegray_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[2], "obikegradh_300x200_P400_-16b.bw"),
                    vxFWriteImageNode(graph, images[3], "obikegradv_300x200_P400_-16b.bw"),
                    vxFWriteImageNode(graph, images[4], "obikemag_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[5], "obikeatan_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[6], "obikeluty_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[7], "obikediff_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[8], "obikethsh_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[9], "obikesums_600x200_P400_16b.bw"),
                    vxFWriteImageNode(graph, images[10], "obikeerod_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[11], "obikedilt_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[12], "obikemed_640x480_P400.pgm"),
                    vxFWriteImageNode(graph, images[13], "obikeavg_640x480_P400.pgm"),
                    vxFWriteImageNode(graph, images[14], "obikediff_300x200_P400_16b.bw"),
                    vxFWriteImageNode(graph, images[15], "obikecanny_640x480_P400.pgm"),
                    vxFWriteImageNode(graph, images[16], "obikeeqhist_640x480_P400.pgm"),
                    vxFWriteImageNode(graph, images[17], "obikegrady8_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[18], "obikeremap_nn_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[19], "obikeremap_bi_300x200_P400.pgm"),
                    vxFWriteImageNode(graph, images[20], "obikeremap_ab_300x200_P400.pgm"),
#endif		    
                };
                vx_print_log((vx_reference)context);
                CHECK_ALL_ITEMS(nodes, i, status, exit);
                if (status == VX_SUCCESS)
                {
                    status = vxVerifyGraph(graph);
                    vx_print_log((vx_reference)graph);
                    if (status == VX_SUCCESS)
                    {
                        status = vxProcessGraph(graph);
                        assert(status == VX_SUCCESS);
                    }
                    else
                    {
                        vx_print_log((vx_reference)graph);
                    }
                    if (status == VX_SUCCESS)
                    {
#if 0
           	        vx_coordinates2d_t min_l, *p_min_l = &min_l;
                        vx_coordinates2d_t max_l, *p_max_l = &max_l;
                        vx_size stride = 0;
                        vx_uint8 min_v = 255;
                        vx_uint8 max_v = 0;

                        vxAccessArrayRange(minLoc, 0, 1, &stride, (void **)&p_min_l, VX_READ_ONLY);
                        vxCommitArrayRange(minLoc, 0, 0, p_min_l);
                        vxAccessArrayRange(maxLoc, 0, 1, &stride, (void **)&p_max_l, VX_READ_ONLY);
                        vxCommitArrayRange(maxLoc, 0, 0, p_max_l);

                        vxReadScalarValue(minVal, &min_v);
                        vxReadScalarValue(maxVal, &max_v);

                        printf("Min Value in AbsDiff = %u, at %d,%d\n", min_v, min_l.x, min_l.y);
                        printf("Max Value in AbsDiff = %u, at %d,%d\n", max_v, max_l.x, max_l.y);

                        vxAccessDistribution(dist, (void **)&histogram, VX_READ_ONLY);
                        for (i = 0; i < windows; i++)
                        {
                            printf("histogram[%u] = %d\n", i, histogram[i]);
                        }
                        vxReadScalarValue(s_mean, &mean);
                        vxReadScalarValue(s_stddev, &stddev);
                        printf("AbsDiff Mean = %lf\n", mean);
                        printf("AbsDiff Stddev = %lf\n", stddev);
                        vxCommitDistribution(dist, histogram);
#endif			
                    }
                    else
                    {
                        printf("Graph failed (%d)\n", status);
                        for (i = 0; i < dimof(nodes); i++)
                        {
                            status = VX_SUCCESS;
                            vxQueryNode(nodes[i], VX_NODE_ATTRIBUTE_STATUS, &status, sizeof(status));
                            if (status != VX_SUCCESS)
                            {
                                printf("nodes[%u] failed with %d\n", i, status);
                            }
                        }
                        status = VX_ERROR_NOT_SUFFICIENT;
                    }
                    for (i = 0; i < dimof(nodes); i++)
                    {
                        vxReleaseNode(&nodes[i]);
                    }
                }
                vxReleaseGraph(&graph);
            }
        }
        for (i = 0; i < dimof(images); i++)
        {
            vxReleaseImage(&images[i]);
        }
exit:
        //vx_print_log((vx_reference)context);
        /* Deregister log callback */
        vxRegisterLogCallback(context, NULL, vx_false_e);
        vxReleaseRemap(&table);
        vxReleaseScalar(&sshift);
        vxReleaseScalar(&snoshift);
        vxReleaseThreshold(&hyst);
        vxReleaseThreshold(&thresh);
        vxReleaseContext(&context);
    }
    return status;
}

/*! \brief The main unit test.
 * \param argc The number of arguments.
 * \param argv The array of arguments.
 * \return vx_status
 * \retval 0 Success.
 * \retval !0 Failure of some sort.
 */
int main(int argc, char *argv[])
{
    vxDDUMPSet(1); // dump debug file	
    vx_status status = vx_mytest_cannyedge(); 
    vxDDUMPSet(0);
    if (status == VX_SUCCESS)
    {
        printf("[PASSED]\n");
    }
    else
    {
        printf("[FAILED]\n");
    }
   
    return 0;
}

