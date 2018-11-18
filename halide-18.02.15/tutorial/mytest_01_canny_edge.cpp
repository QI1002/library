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
    int xm, ym, w, h;

#ifdef BORDER_MODE_REPEAT
    Expr clamped_x = clamp(x, 0, input.width()-1);
    Expr clamped_y = clamp(y, 0, input.height()-1);
    clamped(x, y) = input(clamped_x, clamped_y);    
    xm = ym = 0;
    w = input.width(); h = input.height();
#else
    clamped(x, y) = input(x, y);
    xm = ym = 2;
    w = input.width()-4; h = input.height()-4;    
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
    Buffer<uint16_t> output(w, h); 
    output.set_min(xm, ym);
    //phase.realize(output); // ok 
    //norm.realize(output); // ok 
    suppress.realize(output); // fail if BORDER_MODE_REPEAT off

#if 1
    FILE* wf = fopen("images/suppress.bin", "wb");
    for(int i = 0; i < output.height(); i++) {
        for(int j = 0; j < output.width(); j++) {
            uint16_t v = output(j+xm, i+ym);
            fwrite(&v, sizeof(uint16_t), 1, wf);
        }
    }
    fclose(wf);
#endif

    //11. save the image
    Buffer<uint8_t> picture(w, h); 
    picture.set_min(xm, ym);
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
