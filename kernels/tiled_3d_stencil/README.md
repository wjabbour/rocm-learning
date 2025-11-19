# Overview

A stencil involves performing some mathematical operation both on some entity and its neighbors.

For example, you could imagine a physics simulation where an individual droplet of liquid in a body of water should be influenced by the liquid around it, so you might want to calculate the direction and force of movement of all of the individual droplets which surround this droplet in order to understand the net effect on this droplet - is it pushed downward, upward, etc.

Or perhaps we have a 3D cube split into an arbitrary amount of cells, each cell with a distinct color. We might want to average the color of each cell with its surrounding neighbors.

I now understand that this is quite a fundamental algorithm and can be adapted to accomplish tasks in many different problem domains.

# Learnings

This one took me about 5 days to fully comprehend (1-3 hours each day). The only other kernel I've written was the dead simple one where I launch 1D blocks to add two arrays.

My understanding and ability to conceptualize 3D shapes has greatly improved, now that I've had to spend a lot of time thinking about launching 3D blocks with 3D grids, creating 3D tiles, reading from 1D input and mapping to 3D space. I could feel my brain tingling with growth and confusion :)

Speaking of, this was my first time creating tiles and using LDS. I now understand that LDS is a very fast memory region (I conceptualize that it is similar to the relationship of the CPU and the registers) which sits physically near the compute unit. I understand that each compute unit on the GPU has its own LDS and that the memory allocated by each workgroup in LDS is only visible to that workgroup. That workgroup specificity was a good learning, as I had imagined that if neighboring blocks could cooperatively load and share the data for their shared and overlapping boundaries, we could see a performance improvement in stencil computation. But I've come to understand that the benefit we would reap from this cooperation would be heavily offset by adding much stricter requirements around workgroup scheduling, i.e. CU-affinity for workgroups.

I understand the general pattern of writing a stencil:

- launch blocks of some reasonable thread count, e.g. 512
- create a tile in LDS whose dimensions are the same as the block dimensions, increased by HALO*2 in each dimension (I do realize that HALO*2 may not be applicable to other stencils)
- every thread should load its central cell (itself) into LDS
- threads on the boundaries of blocks should load halos
 
The small block size ensures that each block requests a relatively small and constant size from the LDS, e.g. with HALO=2, 10*10*10 = 1000 bytes of LDS, and LDS on latest gen AMD hardware is 64KB.



