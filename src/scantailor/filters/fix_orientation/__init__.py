"""Fix Orientation filter.

This is the first filter in the processing pipeline. It allows applying
orthogonal rotations (0, 90, 180, 270 degrees) to images.

The rotation is stored as metadata and applied during rendering/output,
not directly to image pixels during this stage.
"""
