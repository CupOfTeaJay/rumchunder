/*
    3D Simplex Noise.
*/

/// Hasher for generating pseudorandom gradient vectors. Or, in accessible
/// terms, pointy arrows that go in different directions :).
///
/// I used an online "Random Integer & Real Number Generator" from
/// https://www.meridianoutpost.com to generate the floats seen in the `dot`
/// and `fract` expressions.
///
/// The hashing algorithm itself I believe was popularized by Ingio Quilez. The
/// idea is to:
///   1. Project the input point onto several unrelated directions
///   2. Feed the projections to `sin`, which exhibits chaotic behavior for 
///      large arguments
///        a. I'm definitely not a signal processing person, but I found a
///           Reddit comment that says this exploits "aliasing"
///   3. Amplify and take the fractional part to obtain values in [0, 1)
///   4. Linearly remap the result to [-1, 1]
///
/// I also added a `seed` to perturb the `sin` call a bit. This might not be the
/// best way to introduce seeded noise, but I think it should work.
fn hash(point: vec3f, seed: u32) -> vec3f {
    let projections: vec3f = vec3f(
        dot(point, vec3f(313.785, 201.681, 61.266)),
        dot(point, vec3f(57.572, 321.757, 75.291)),
        dot(point, vec3f(222.241, 251.138, 198.24)),
    );
    return 2.0*fract(3913.188*sin(projections + vec3f(f32(seed)))) - 1.0;
}

/// Simplex noise algorithm in three dimensions. A `seed` may be provided to
/// get reproducible results.
///
/// The implementation is intentionally verbose. I am not a graphics person at
/// all. This was mostly an exercise to understand (at least partially) the
/// steps involved in the algorithm. It's likely that some snippets might not
/// be altogether correct. Feedback is always welcomed and much appreciated.
///
/// I frequently referenced the following:
///   * Simplex Noise & Perlin Noise (https://compute.toys/view/1896)
///       * Anonymous
///   * Simplex Noise (https://compute.toys/view/16)
///       * David A Roberts
///   * Simplex Noise Demystified
///     (https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/simplexnoise.pdf)
///       * Stefan Gustavson
fn simplex_3d(point: vec3f, seed: u32) -> f32 {
    // Unlike classical Perlin noise, which is derived from a space tesselated
    // by hypercubes, Simplex noise operates upon a space tesselated by
    // "simplices". In three dimensions this simplex takes the form of a
    // tetrahedron.
    //
    // Again, much like Perlin noise, we need to figure out where the input
    // `point` lies in this space. To do this we can leverage the fact that
    // Simplex grids can be naturally expressed as "skewed" hypercube grids.
    // This is done by translating every point in a hypercube grid along that
    // diagonal where (x, y, z) increases uniformly.
    let skew_factor: f32 = 1.0/3.0;
    let skewed_point: vec3f = point + dot(point, vec3f(skew_factor));

    // Next, we simply drop the fractional parts. This effectively determines
    // the vertex of the rhombohedron (and consequently, the tetrahedron as
    // we will see later) that encloses our point.
    let vertex_0: vec3f = floor(skewed_point);

    // Now we need to figure out where our point is "locally" within its
    // rhombohedron. To make a valid calculation, the rhombohedron vertex
    // needs to be mapped back into the hypercube space by way of an offset.
    let unskew_factor: f32 = 1.0 / 6.0;
    let unskew_offset: vec3f = vec3f(dot(vertex_0, vec3f(unskew_factor)));
    let local_point: vec3f = point - vertex_0 + unskew_offset;

    // A rhombohedron in this space is composed of six tetrahedra. Therefore,
    // we can now find our simplex by determining which of the six `local_point`
    // falls into. To do this branchless-ly we'll encode some bools such that:
    //   * flags.x = 1 if x >= y else 0
    //   * flags.y = 1 if y >= z else 0
    //   * flags.z = 1 if z >= x else 0
    let flags: vec3f = step(vec3f(0.0), local_point - local_point.yzx);

    // Conceptually, we can think of `vertex_0` as now the vertex of one of the
    // tetrahedra that compose the original rhombohedron. The flags we just
    // derived allow us to get integer offsets for two out of the remaining
    // three vertices.
    let vertex_1_offset: vec3f = flags*(1.0 - flags.zxy);
    let vertex_2_offset: vec3f = 1.0 - flags.zxy*(1.0 - flags);
    let vertex_3_offset: vec3f = vec3f(1.0); // Trivial.

    // At this point we have five points to work with:
    //   * The input `point`'s "local" position within a greater simplex
    //   * The four vertices of that same simplex
    //
    // In a perlin-esque fashion, we'll want to calculate the displacement of
    // the local position with respect to each of the tetrahedron's vertices.
    // Note that the offsets are still in the skewed basis, so we'll have to
    // correct for that as well.
    let disp_0: vec3f = local_point;
    let disp_1: vec3f = local_point - vertex_1_offset + unskew_factor;
    let disp_2: vec3f = local_point - vertex_2_offset + 2.0*unskew_factor;
    let disp_3: vec3f = local_point - vertex_3_offset + 3.0*unskew_factor;

    // Squaring each of the displacements gives us an idea of how far away we
    // are from the vertices. Near a vertex, components of `disp_squared` will
    // approach zero.
    let disp_squared: vec4f = vec4f(
        dot(disp_0, disp_0),
        dot(disp_1, disp_1),
        dot(disp_2, disp_2),
        dot(disp_3, disp_3),
    );

    // The "influence" each vertex surflet will have on our point is dependent
    // on the point's distance from each, effectively `disp_squared`. We want
    // the influences to be "local". Thus, we can clamp contributions greater
    // than a constant radius of `0.6`.
    //
    // To visualize this, we might imagine each vertex as the center of a
    // spherical "bubble" with a squared radius of `0.6`. Points that fall
    // outside of this bubble will not be influenced by the vertex's kernel. In
    // other words, its influence has attenuated completely.
    let disp_squared_clamped: vec4f = max(0.6 - disp_squared, vec4f(0.0));

    // Raising the influences to the fourth power transforms the attenuation
    // behavior from sharp and linear to smooth. We now have our surflet
    // kernels.
    let surflet_weights: vec4f = pow(disp_squared_clamped, vec4f(4.0));

    // This is the typical "pseudorandom gradient vector" part of the algorithm.
    // We're assigning, well... a pseudorandom gradient vector to each of the
    // vertices:
    //   * vertex_0 = vertex_0
    //   * vertex_1 = vertex_0 + vertex_1_offset
    //   * vertex_2 = vertex_0 + vertex_2_offset
    //   * vertex_3 = vertex_0 + vertex_3_offset
    //
    // This is essentially the source of randomness, or noise. Much like how our
    // displacement vectors `disp_n` are (dropping the mathematical rigor) just
    // arrows pointing from a vertex to `local_point`, these `gradient_n`
    // vectors are simply arrows pointing from a vertex to a random direction.
    // The only difference being that the gradient vectors are normalized.
    let gradient_0: vec3f = normalize(hash(vertex_0, seed));
    let gradient_1: vec3f = normalize(hash(vertex_0 + vertex_1_offset, seed));
    let gradient_2: vec3f = normalize(hash(vertex_0 + vertex_2_offset, seed));
    let gradient_3: vec3f = normalize(hash(vertex_0 + vertex_3_offset, seed));

    // Almost there. Now we want to get the projection of each of the gradient
    // vectors onto the displacement vectors to get a measure of orthogonality.
    let projection_0: f32 = dot(gradient_0, disp_0);
    let projection_1: f32 = dot(gradient_1, disp_1);
    let projection_2: f32 = dot(gradient_2, disp_2);
    let projection_3: f32 = dot(gradient_3, disp_3);

    // Finally, we sample a noise value. At the end of the day, it is the
    // projections we just calculated biased with respect to the influence of
    // each surflet kernel.
    let sample: f32 =
        surflet_weights.x*projection_0 +
        surflet_weights.y*projection_1 +
        surflet_weights.z*projection_2 +
        surflet_weights.w*projection_3;

    // A few things to note before returning the `sample`:
    //   * Modelling the surflet influences to the 4th-order greatly reduces
    //     the sampled values
    //   * We can counteract the above by multiplying by some `gain` (feel free
    //     to play around with this value to see the effect it has on the final
    //     result)
    //   * Gained samples also fall within the range [-1, 1], so we do a linear
    //     transformation by multiplying and adding 0.5 to map things to [0, 1].
    let gain: f32 = 32.0;
    return sample*gain*0.5 + 0.5;
}

