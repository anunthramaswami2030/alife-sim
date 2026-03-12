import numpy as np
from scipy import ndimage

# Size of the binary mask (8x8 grid of voxels)
MASK_DIM = 8 

# Scale of the robot (edge length of a voxel)
# NOTE: this is very important as the simulator physics are configured to use this scale, more or less.
SCALE = 0.1 

def load_robots(num_robots):
    return [sample_robot() for _ in range(num_robots)]

# Randomly sample a binary mask of size MASK_DIM x MASK_DIM
# Convert the binary mask to a mass-spring robot geometry
# The parameter p is by default set to 0.55, which is the probability of a voxel being filled.
# This is a manually tuned value that seems to produce a variety of different robot geometries.
def sample_robot(p=0.55):
    mask = sample_mask(p)
    masses, springs = mask_to_robot(mask)
    masses = masses * SCALE # NOTE: scale of the robot geometry is KEY to stable simulation!
    return {
        "n_masses": masses.shape[0],
        "n_springs": springs.shape[0],
        "masses": masses,
        "springs": springs,
        "mask": mask
    }

# Convert a voxel position to a list of mass coordinates
# Each voxel has a mass located at each of its four corners
def voxel_to_masses(row, col):
    return [
        [row, col],
        [row, col+1],
        [row+1, col],
        [row+1, col+1],
    ]

# Convert a binary mask to a mass-spring robot geometry
# Each voxel is represented by 4 masses and 6 springs
# Masses are located at the corners of the voxel
# Springs connect adjacent masses along the edges and diagonals of the voxel
def mask_to_robot(mask):
    spring_connections = [
        [0, 1], # bottom left (bl) to bottom right (br)
        [0, 2], # bl to top left (tl)
        [1, 3], # br to top right (tr)
        [2, 3], # tl to tr
        [0, 3], # bl to tr
        [1, 2], # br to tl
    ]
    masses = []
    springs = []
    rows, cols = np.where(mask)
    n_voxels = len(rows)
    for i in range(n_voxels):
        row = rows[i]
        col = cols[i]
        coords = voxel_to_masses(row, col)
        for c in coords:
            if c not in masses: # NOTE: make sure to avoid duplicates!
                masses.append(c)
        for a, b, in spring_connections:
            ca = coords[a]
            cb = coords[b]
            ia = masses.index(ca)
            ib = masses.index(cb)
            s = [min(ia, ib), max(ia, ib)]
            if s not in springs: # NOTE: make sure to avoid duplicates!
                springs.append(s)
    masses = np.array(masses, dtype=np.float32) # Numpy array of shape (n_masses, 2)
    springs = np.array(springs, dtype=np.int32) # Numpy array of shape (n_springs, 2)
    return masses, springs

# Sample a binary mask of size MASK_DIM x MASK_DIM
# Select the largest connected component in the mask
# Zero out the rest of the mask
# Shift the largest component to the bottom left corner of the mask
def sample_mask(p):
    while True:
        mask = np.random.uniform(0.0, 1.0, size=(MASK_DIM, MASK_DIM)) < p
        mask = canonicalize_mask(mask)
        if mask.any():
            return mask

def boundary_candidates(mask_bool: np.ndarray):
    struct = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=bool)

    dil = ndimage.binary_dilation(mask_bool, structure=struct)
    ero = ndimage.binary_erosion(mask_bool, structure=struct)

    add = np.argwhere(dil & ~mask_bool)
    rem = np.argwhere(mask_bool & ~ero)
    return add, rem

def canonicalize_mask(mask):
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=int)
    component_sizes = ndimage.sum(mask, labeled,
                                  range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = (labeled == largest_component)
    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    component = mask[r0:r1+1, c0:c1+1]
    out = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    h, w = component.shape
    out[MASK_DIM - h:MASK_DIM, 0:w] = component.astype(int)
    return out

def mutate_mask(mask: np.ndarray,
                n_edits: int = 1,
                add_prob: float = 0.6,
                ensure_nonempty: bool = True,
                rng: np.random.Generator | None = None) -> np.ndarray:
    
    rng = rng or np.random.default_rng()
    m = mask.astype(bool).copy()
    for _ in range(n_edits):
        add_cand, rem_cand = boundary_candidates(m)
        do_add = (rng.random() < add_prob)
        if do_add and len(add_cand) > 0:
            r, c = add_cand[rng.integers(len(add_cand))]
            m[r, c] = True
        elif (not do_add) and len(rem_cand) > 0:
            r, c = rem_cand[rng.integers(len(rem_cand))]
            m[r, c] = False
        else:
            if len(add_cand) > 0:
                r, c = add_cand[rng.integers(len(add_cand))]
                m[r, c] = True
            elif len(rem_cand) > 0:
                r, c = rem_cand[rng.integers(len(rem_cand))]
                m[r, c] = False          
    out = canonicalize_mask(m)
    if ensure_nonempty and not out.any():
        return mask
    return out

def crossover(parent_a, parent_b, rng, min_fill=3, max_tries=30):
    A = parent_a.astype(bool)
    B = parent_b.astype(bool)

    # Precompute boundary candidates (you already have this helper)
    addA, remA = boundary_candidates(A)   # add candidates touch A
    addB, remB = boundary_candidates(B)

    # If either parent is empty-ish, fall back
    if A.sum() == 0 or B.sum() == 0:
        return canonicalize_mask(A)

    for _ in range(max_tries):
        child = A.copy()

        # pick a random rectangle in B
        r0 = rng.integers(0, MASK_DIM)
        r1 = rng.integers(r0 + 1, MASK_DIM + 1)
        c0 = rng.integers(0, MASK_DIM)
        c1 = rng.integers(c0 + 1, MASK_DIM + 1)

        patch = B[r0:r1, c0:c1]
        if patch.sum() == 0:
            continue

        # choose a target location in A near its boundary so we "attach" the patch
        if len(addA) == 0:
            # A is full or weird; just paste at same coords
            tr0, tc0 = r0, c0
        else:
            tr, tc = addA[rng.integers(len(addA))]
            # align patch roughly around chosen attachment point
            tr0 = np.clip(tr - (r1 - r0)//2, 0, MASK_DIM - (r1 - r0))
            tc0 = np.clip(tc - (c1 - c0)//2, 0, MASK_DIM - (c1 - c0))

        tr1 = tr0 + (r1 - r0)
        tc1 = tc0 + (c1 - c0)

        child[tr0:tr1, tc0:tc1] = patch | child[tr0:tr1, tc0:tc1]

        out = canonicalize_mask(child)
        if out.sum() >= min_fill:
            return out

    # fallback: simple union tends to be safer than random rectangle overwrite
    return canonicalize_mask(A | B)