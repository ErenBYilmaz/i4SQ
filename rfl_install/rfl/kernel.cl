// BEWARE: The used literals 'x', 'y' and 'z' don't correspond to the actual
// axis of the image, since the first dimension internally corresponds to the
// 'z' axis.

#define FLAT_INDEX_3D(position, size) \
    (((position).s0 * (size).s1 + (position).s1) * (size).s2 + (position).s2)
#define FLAT_INDEX_4D(position, size) \
    (FLAT_INDEX_3D(position, size) * (size).s3 + (position).s3)

#define THREED_INDEX_0(index, size) ((index) / ((size).s1 * (size).s2))
#define THREED_INDEX_1(index, size) (((index) / (size).s2) % (size).s1)
#define THREED_INDEX_2(index, size) ((index) % (size).s2)


inline void atomic_add_global(volatile global float *source, const float operand);


kernel void predict(constant float *image,
                    const int4 size,
                    constant int *offsets1,
                    constant int *offsets2,
                    constant int *children_left,
                    constant int *children_right,
                    constant int *split_features,
                    constant float *split_thresholds,
                    constant float *values,
                    const float oob_value,
                    global float *out)
{
    const int index = get_global_id(0);

    // the position inside the image we want to predict the value for and
    // the corresponding value
    const int4 position = {THREED_INDEX_0(index, size),
                           THREED_INDEX_1(index, size),
                           THREED_INDEX_2(index, size),
                           0};

    // make sure the pixel we are processing is actually in the image
    if (position.s0 >= size.s0 || position.s1 >= size.s1 || position.s2 >= size.s2)
        return;

    // index of the tree node we are currently at
    int node = 0;

    do {
        const int feature_index = split_features[node] * 4;

        // <kernel>:49:47: error: Explicit cast from address space "constant" to address space "private" is not allowed
        //const int4 offset2 = * (const int4 *) (&offsets2[feature_index]);
        const int4 offset1 = {offsets1[feature_index],
                              offsets1[feature_index + 1],
                              offsets1[feature_index + 2],
                              offsets1[feature_index + 3]};
        const int4 offset2 = {offsets2[feature_index],
                              offsets2[feature_index + 1],
                              offsets2[feature_index + 2],
                              offsets2[feature_index + 3]};

        const int4 feature_position1 = position + offset1;
        const int4 feature_position2 = position + offset2;

        float feature_value;
        if (feature_position1.s0 < 0 || feature_position1.s0 >= size.s0 ||
            feature_position1.s1 < 0 || feature_position1.s1 >= size.s1 ||
            feature_position1.s2 < 0 || feature_position1.s2 >= size.s2 ||
            feature_position2.s0 < 0 || feature_position2.s0 >= size.s0 ||
            feature_position2.s1 < 0 || feature_position2.s1 >= size.s1 ||
            feature_position2.s2 < 0 || feature_position2.s2 >= size.s2)
        {
            feature_value = oob_value;
        } else {
            feature_value = image[FLAT_INDEX_4D(feature_position2, size)] -
                image[FLAT_INDEX_4D(feature_position1, size)];
        }

        if (feature_value <= split_thresholds[node]) {
            node = children_left[node];
        } else {
            node = children_right[node];
        }
    } while (children_left[node] != -1);

    atomic_add_global(&out[FLAT_INDEX_3D(position, size)], values[node]);
}


// helper function to perform an atomic add of float values
inline void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal)
        != prevVal.intVal);
}
