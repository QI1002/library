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
using namespace Halide;

// Include some support code for loading pngs.
#include "halide_image_io.h"
using namespace Halide::Tools;

#define BORDER_MODE_REPEAT //  default: skip border 
#define TARGET_OPENCL // default: openCL
//#define NORMAL_L2 // default: L2

int main(int argc, char **argv) {

    //01. open file and how it's property and define common variables
    Buffer<uint8_t> input = load_image("images/bike.png");
    printf("input image = %d %d %d\n", input.width(), input.height(), input.channels());
    Var x, y, xg, yg;

    //02. define clamp function for border mode = repeat
    Func clamped;
    Expr clamped_x = clamp(x, 0, input.width()-1);
    Expr clamped_y = clamp(y, 0, input.height()-1);
    clamped(x, y) = input(clamped_x, clamped_y);    
    
    //03. define cast uint16 filter to avoid overflow or underflow
    Func input16;
    input16(x, y) = cast<uint16_t>(clamped(x, y));

    //04. define gradient x,y filter 
    Func gx, gy;
    gx(x, y) = (input16(x+1, y-1) + 2*input16(x+1, y) + input16(x+1, y+1) - 
		input16(x-1, y-1) - 2*input16(x-1, y) - input16(x-1, y+1));
    gy(x, y) = (input16(x-1, y+1) + 2*input16(x, y+1) + input16(x+1, y+1) - 
		input16(x-1, y-1) - 2*input16(x, y-1) - input16(x+1, y-1));

    //05. define normalize filter
    Func norm;
#ifdef NORMAL_L2
    norm(x, y) = cast<uint8_t>(gx(x, y) * gx(x, y) + gy(x, y) * gy(x, y));  
#else    
    Expr absx = select(gx(x, y) >= 0, gx(x, y), -gx(x, y));
    Expr absy = select(gy(x, y) >= 0, gy(x, y), -gy(x, y));
    norm(x, y) = cast<uint8_t>(absx + absy); 
#endif

    //06. realize 
    Buffer<uint8_t> output = norm.realize(input.width(), input.height());

    //07. save the image
    save_image(output, "images/bike1.png");
    printf("Success!\n");
    
    return 0;
}
