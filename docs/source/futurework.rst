Future work
===========

Here are some ideas to develope this project further

* Varying radius of spheres in the particle cluster: This can quite easily be implemented, at least to let the radius increase and overlap other subspheres. Since the distance between the centerpoints of subspheres can not be changed a radius less than that distance would disconnect them in the current implementation.


* Place subspheres in a continous matter rather than discrete (left, right, up, down). This would require a complete rework of at least :meth:`utils.gen_polydata`.


* Use depth pictures instead of simply black and white pictures to incorperate more informations in the pictures. This would not be that hard since there exists functions in pyvista for this. Checkout their tutorial for `this <https://docs.pyvista.org/examples/02-plot/image_depth>`_.


* Currently 3 images as fed as separate channels into a CNN, we have seen work that combines 3 images into a single image as feed this image with good results, could be interesting to  evaluate this approach. Check out this `paper <https://arxiv.org/pdf/2306.06110>`_.


* The size of the dataset is limited and limits the performance, more data = better. 

