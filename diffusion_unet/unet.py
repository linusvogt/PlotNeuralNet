import sys
sys.path.append('../')
from pycore.tikzeng import *

unet = [
    to_head( '..' ),    # preamble
    to_cor(),           # define colors
    to_begin(),         # begin document

    ### -- Input
    to_input("/home/linus/PlotNeuralNet/diffusion_unet/input.png", name='input'),
    
    ### -- E1; note: D=(H, W)64
    to_ConvConvRelu(
        name="e1",
        s_filer='D', n_filer=(32, 32),
        offset="(1,0,0)", to="(0,0,0)",
        height=25, depth=50, width=(2, 2),
        caption='E1'
    ),

    ### -- E2
    to_ConvConvRelu(
        name="e2",
        s_filer='D/2', n_filer=(64, 64),
        offset="(2,-4,0)", to="(e1-east)",
        height=25, depth=50, width=(4, 4),
        caption='E2'
    ),
    to_connection("e1", "e2"),

    ### -- E3
    to_ConvConvRelu(
        name="e3",
        s_filer='D/4', n_filer=(128, 128),
        offset="(2,-4,0)", to="(e2-east)",
        height=25, depth=50, width=(8, 8),
        caption='E3'
    ),
    to_connection("e2", "e3"),

    ### -- Bottleneck
    to_ConvConvRelu(
        name="bottleneck",
        s_filer='D/4', n_filer=(128, 128),
        offset="(5,0,0)", to="(e3-east)",
        height=25, depth=50, width=(8, 8),
        caption='Bottleneck'
    ),
    to_connection("e3", "bottleneck"),

    ### -- D3
    to_ConvConvRelu(
        name="d3",
        s_filer='D/4', n_filer=(128, 128),
        offset="(5,0,0)", to="(bottleneck-east)",
        height=25, depth=50, width=(8, 8),
        caption='D3'
    ),
    to_connection("bottleneck", "d3"),

    ### -- D2
    to_ConvConvRelu(
        name="d2",
        s_filer='D/2', n_filer=(64, 64),
        offset="(5,4,0)", to="(d3-east)",
        height=25, depth=50, width=(4, 4),
        caption='D2'
    ),
    to_connection("d3", "d2"),

    ### -- D1
    to_ConvConvRelu(
        name="d1",
        s_filer='D', n_filer=(32, 32),
        offset="(5,4,0)", to="(d2-east)",
        height=25, depth=50, width=(2, 2),
        caption='D1'
    ),
    to_connection("d2", "d1"),

    to_skip("e1", "d1"),
    to_skip("e2", "d2"),
    to_skip("e3", "d3"),


    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(unet, namefile + '.tex' )

if __name__ == '__main__':
    main()
