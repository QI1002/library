// Halide tutorial lesson 2: Processing images

// This lesson demonstrates how to implement vx canny edge
// them.

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_mytest_01_canny_edge
// in a shell with the current directory at the top of the halide
// source tree.

// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"
#include <stdio.h>
using namespace Halide;

// Include some support code for loading pngs.
#include "halide_image_io.h"
using namespace Halide::Tools;

//A definition for TAU, or 2*PI.
#define TAU 6.28318530717958647692f
//#define BORDER_MODE_REPEAT //  default: skip border 
//#define TARGET_OPENCL // default: CPU
//#define TARGET_CUDA // default: CPU 
//#define NORMAL_L2 // default: L2

bool have_opencl_or_metal();
int main(int argc, char **argv) {
    //01. open file and how it's property and define common variables
    Buffer<uint8_t> input = load_image("images/bike.png");
    printf("input image = %d %d %d\n", input.width(), input.height(), input.channels());
    Var x, y;

    //02. define clamp function for border mode = repeat
    Func clamped;
    int xm1, ym1, xm2, ym2, w1, h1, w2, h2;

#ifdef BORDER_MODE_REPEAT
    Expr clamped_x = clamp(x, 0, input.width()-1);
    Expr clamped_y = clamp(y, 0, input.height()-1);
    clamped(x, y) = input(clamped_x, clamped_y);    
    xm1 = ym1 = xm2 = ym2 = 0;
    w1 = input.width(); h1 = input.height();
    w2 = input.width(); h2 = input.height();
#else
    clamped(x, y) = input(x, y);
    xm1 = ym1 = 1; xm2 = ym2 = 2;
    w1 = input.width()-2; h1 = input.height()-2;    
    w2 = input.width()-4; h2 = input.height()-4;    
#endif

    //03. define cast uint16 filter to avoid overflow or underflow
    Func input16;
    input16(x, y) = cast<int16_t>(clamped(x, y));

    //04. define gradient x,y filter 
    Func gx, gy;
    gx(x, y) = (input16(x-1, y-1) + 2*input16(x-1, y) + input16(x-1, y+1) - 
		input16(x+1, y-1) - 2*input16(x+1, y) - input16(x+1, y+1));
    gy(x, y) = (input16(x-1, y+1) + 2*input16(x, y+1) + input16(x+1, y+1) - 
		input16(x-1, y-1) - 2*input16(x, y-1) - input16(x+1, y-1));

    //05. define normalize filter
    Func norm;
#ifdef NORMAL_L2
    norm(x, y) = cast<uint16_t>(sqrt((gx(x, y) * gx(x, y) + gy(x, y) * gy(x, y))) + 0.5f);  
#else    
    norm(x, y) = cast<uint16_t>(abs(gx(x,y)) + abs(gy(x, y))); 
#endif

    //06. define phase filter 
    Func phase;
    Expr angle = atan2(cast<float>(gy(x, y)), cast<float>(gx(x, y)));    
    angle = select(angle < 0.0f, angle + TAU, angle)/TAU;
    angle = cast<uint8_t>(cast<uint32_t>(angle*256 + 0.5f) & 0xFF);
    phase(x, y) = angle;    

    //07. define the clamped normal  
    Func normc;
#ifdef BORDER_MODE_REPEAT
    normc(x, y) = cast<uint16_t>(norm(clamped_x, clamped_y));    
#else
    normc(x, y) = cast<uint16_t>(norm(x, y));    
#endif

    //08. non-max surpression and use mutliple switch
    Func suppress;
    Expr ix = (((cast<uint16_t>(phase(x, y)) + 16)/32) & 0x3)+5;
    suppress(x,y) = select(ix == 5, 
                           select(normc(x, y) > normc(x+1, y) && 
                                  normc(x, y) > normc(x-1, y), normc(x, y), 0), 
                    select(ix == 6, 
                           select(normc(x, y) > normc(x+1, y-1) && 
                                  normc(x, y) > normc(x-1, y+1), normc(x, y), 0),
                    select(ix == 7, 
                           select(normc(x, y) > normc(x, y-1) && 
                                  normc(x, y) > normc(x, y+1), normc(x, y), 0),
                    select(ix == 8, 
                           select(normc(x, y) > normc(x+1, y+1) && 
                                  normc(x, y) > normc(x-1, y-1), normc(x, y), 0), 0))));

    //09. possible to change to other target 
    Target target = get_host_target();
    if (have_opencl_or_metal()) {
#ifdef TARGET_OPENCL	    
        target.set_feature(Target::OpenCL);
#endif	
#ifdef TARGET_CUDA
	target.set_feature(Target::CUDA);
#endif	
    }
    // not let the jit compile impact performance measure
    suppress.compile_jit(target);
    
    //10. realize
    Buffer<int16_t> outputx(w1, h1); 
    outputx.set_min(xm1, ym1);
    gx.realize(outputx);
    FILE* wfx = fopen("images/gx.bin", "wb");
    for(int i = 0; i < outputx.height(); i++) {
        for(int j = 0; j < outputx.width(); j++) {
            int16_t v = outputx(j+xm1, i+ym1);
            fwrite(&v, sizeof(int16_t), 1, wfx);
        }
    }
    fclose(wfx);

    Buffer<int16_t> outputy(w1, h1); 
    outputy.set_min(xm1, ym1);
    gy.realize(outputy);
    FILE* wfy = fopen("images/gy.bin", "wb");
    for(int i = 0; i < outputy.height(); i++) {
        for(int j = 0; j < outputy.width(); j++) {
            int16_t v = outputy(j+xm1, i+ym1);
            fwrite(&v, sizeof(int16_t), 1, wfy);
        }
    }
    fclose(wfy);
    
    Buffer<uint16_t> outputn(w1, h1); 
    outputn.set_min(xm1, ym1);
    norm.realize(outputn);
    FILE* wfn = fopen("images/norm.bin", "wb");
    for(int i = 0; i < outputn.height(); i++) {
        for(int j = 0; j < outputn.width(); j++) {
            uint16_t v = outputn(j+xm1, i+ym1);
            fwrite(&v, sizeof(uint16_t), 1, wfn);
        }
    }
    fclose(wfn);

    Buffer<uint8_t> outputp(w1, h1); 
    outputp.set_min(xm1, ym1);
    phase.realize(outputp);
    FILE* wfp = fopen("images/phase.bin", "wb");
    for(int i = 0; i < outputp.height(); i++) {
        for(int j = 0; j < outputp.width(); j++) {
            uint8_t v = outputp(j+xm1, i+ym1);
            fwrite(&v, sizeof(uint8_t), 1, wfp);
        }
    }
    fclose(wfp);

    Buffer<uint16_t> outputs(w2, h2); 
    outputs.set_min(xm2, ym2);
    suppress.realize(outputs);
    FILE* wfs = fopen("images/suppress.bin", "wb");
    for(int i = 0; i < outputs.height(); i++) {
        for(int j = 0; j < outputs.width(); j++) {
            uint16_t v = outputs(j+xm2, i+ym2);
            fwrite(&v, sizeof(uint16_t), 1, wfs);
        }
    }
    fclose(wfs);

    //11. save the image
    Buffer<uint8_t> picture(w2, h2); 
    picture.set_min(xm2, ym2);
    Func result;
    result(x, y) = cast<uint8_t>(suppress(x, y));
    result.realize(picture);
    save_image(picture, "images/bike1.png");
    printf("Success!\n");
    
    return 0;
}

// A helper function to check if OpenCL seems to exist on this machine.

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

bool have_opencl_or_metal() {
#ifdef _WIN32
    return LoadLibrary("OpenCL.dll") != NULL;
#elif __APPLE__
    return dlopen("/System/Library/Frameworks/Metal.framework/Versions/Current/Metal", RTLD_LAZY) != NULL;
#else
    return dlopen("libOpenCL.so", RTLD_LAZY) != NULL;
#endif
}
