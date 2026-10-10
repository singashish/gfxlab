#pragma once

#ifdef __CUDACC__
    #define GFX_HD __host__ __device__
#else
    #define GFX_HD
#endif
