# Overview

A stencil involves performing some mathematical operation both on some entity and its neighbors.

For example, you could imagine a physics simulation where an individual droplet of liquid in a body of water should be influenced by the liquid around it, so you might want to calculate the direction and force of movement of all of the individual droplets which surround this droplet in order to understand the net effect on this droplet - is it pushed downward, upward, etc.

Or perhaps we have a 3D cube split into an arbitrary amount of cells, each cell with a distinct color. We might want to average the color of each cell with its surrounding neighbors.

I now understand that this is quite a fundamental algorithm and can be adapted to accomplish tasks in many different problem domains.
