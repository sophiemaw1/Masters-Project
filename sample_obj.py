import numpy as np
import jax.numpy as jnp
import trimesh
import jax
from jax import random
from tqdm import tqdm


class SampleObj():
    def __init__(self, obj_file_path, num_samples, num_one_rings=None):
        self.num_samples = num_samples
        self.mesh = trimesh.load(obj_file_path)
        self.vertices = jnp.array(self.mesh.vertices)
        self.num_vertices = len(self.vertices)
        self.faces = jnp.array(self.mesh.faces)
        self.texture_image = self.mesh.visual.material.image
        self.texture_image_np = np.array(self.texture_image)
        self.uv_coordinates = jnp.array(self.mesh.visual.uv)
        self.uv_coordinates = (self.uv_coordinates - jnp.min(self.uv_coordinates, axis=0)) / (jnp.max(self.uv_coordinates, axis=0) - jnp.min(self.uv_coordinates, axis=0))


        if num_one_rings is not None:
            self.num_vertices = num_one_rings

        # Preallocate arrays
        self.pix_tri = jnp.zeros((self.num_vertices, num_samples, 3), dtype=jnp.int32)
        self.pix_logs = jnp.zeros((self.num_vertices, num_samples, 3, 2), dtype=jnp.float32)
        self.ring_logs = jnp.zeros((self.num_vertices, num_samples, 2), dtype=jnp.float32)
        self.ring_pix = jnp.zeros((self.num_vertices, num_samples, 3), dtype=jnp.uint32)

    def sample(self):
        key = random.PRNGKey(0)
        for v in range(self.num_vertices):
            v_pos = self.uv_coordinates[v, :2]
            one_ring_faces = self.get_one_ring_faces(v)

            for sample in range(self.num_samples):
                key, subkey = random.split(key)
                s_pos, s_ni, s_rgb = self.randomly_get_sample(subkey, one_ring_faces)
                s_pos = s_pos[:2]

                self.ring_pix = self.update_array(self.ring_pix, v, sample, value=s_rgb)
                self.ring_logs = self.update_array(self.ring_logs, v, sample, value=self.calculate_logarithm(s_pos, v_pos))
                self.pix_tri = self.update_array(self.pix_tri, v, sample, value=s_ni)

                for i, face_index in enumerate(s_ni):
                    face_vertex = self.uv_coordinates[face_index, :2]
                    self.pix_logs = self.update_array(self.pix_logs, v, sample, i, value=self.calculate_logarithm(s_pos, face_vertex))

    def update_array(self, arr, *indices, value):
        return arr.at[indices].set(value)

    def calculate_logarithm(self, coord1, coord2):
        diff = jnp.abs(coord2 - coord1)
        return jnp.log(diff + 1e-9)  # Add a small value to avoid log(0)

    def get_one_ring_faces(self, v_index):
        # Get all faces containing the vertex
        return jnp.where(jnp.any(self.faces == v_index, axis=1))[0]

    def randomly_get_sample(self, key, one_ring_faces):
        face_areas = jnp.array([self.mesh.area_faces[face] for face in one_ring_faces])
        probabilities = face_areas / jnp.sum(face_areas)

        selected_face_index = random.choice(key, one_ring_faces.shape[0], p=probabilities)
        selected_face = self.faces[one_ring_faces[selected_face_index]]

        sample_uv = random.uniform(key, (2,))
        u, v = jnp.sqrt(sample_uv[0]), sample_uv[1]

        vertices = self.vertices[selected_face]
        sampled_point = (1 - u) * vertices[0] + u * (1 - v) * vertices[1] + u * v * vertices[2]
        sampled_point = sampled_point.at[2].set(0)

        uv_coords_face = self.uv_coordinates[selected_face]
        uv_coordinates_sampled_point = (1 - u - v) * uv_coords_face[0] + u * uv_coords_face[1] + v * uv_coords_face[2]

        neighbor_vertex_indices = self.faces[one_ring_faces[selected_face_index]]
        sampled_color = self.sample_texture(uv_coordinates_sampled_point)

        return uv_coordinates_sampled_point, neighbor_vertex_indices, sampled_color

    def sample_texture(self, uv_coordinates):
        image = self.texture_image_np
        height, width = image.shape[:2]
        pixel_coordinates = (uv_coordinates * jnp.array([width, height])).astype(int)
        pixel_coordinates = jnp.clip(pixel_coordinates, 0, jnp.array([width - 1, height - 1]))

        sampled_colors = image[pixel_coordinates[1], pixel_coordinates[0]]
        return jnp.squeeze(sampled_colors)

    def get_data(self):
        return self.ring_logs, self.ring_pix, self.pix_tri, self.pix_logs

    def save_data_to_csv(self, directory):
        np.savetxt(f"{directory}/ring_logs.csv", self.ring_logs.reshape(-1, self.ring_logs.shape[-1]), delimiter=",")
        np.savetxt(f"{directory}/ring_pix.csv", self.ring_pix.reshape(-1, self.ring_pix.shape[-1]), delimiter=",")
        np.savetxt(f"{directory}/pix_tri.csv", self.pix_tri.reshape(-1, self.pix_tri.shape[-1]), delimiter=",")
        np.savetxt(f"{directory}/pix_logs.csv", self.pix_logs.reshape(-1, self.pix_logs.shape[-1] * self.pix_logs.shape[-2]), delimiter=",")

    def load_data_from_csv(self, directory):
        self.ring_logs = np.loadtxt(f"{directory}/ring_logs.csv", delimiter=",").reshape(self.num_vertices, self.num_samples, -1)
        self.ring_pix = np.loadtxt(f"{directory}/ring_pix.csv", delimiter=",").reshape(self.num_vertices, self.num_samples, -1)
        self.pix_tri = np.loadtxt(f"{directory}/pix_tri.csv", delimiter=",").reshape(self.num_vertices, self.num_samples, -1)
        self.pix_logs = np.loadtxt(f"{directory}/pix_logs.csv", delimiter=",").reshape(self.num_vertices, self.num_samples, -1, 2)

    def get_uv_coords(self):
        return self.uv_coordinates


def get_flvae_data_from_obj(file_path, num_one_rings, num_samples):
    sampler = SampleObj(file_path, num_samples, num_one_rings)
    sampler.sample()
    ring_logs, ring_pix, pix_tri, pix_logs = sampler.get_data()
    return ring_logs,ring_pix,pix_tri,pix_logs


def get_flvae_data_from_objs(num_one_rings, num_objs, num_samples):
    ring_logs_list = []
    ring_pix_list = []
    pix_tri_list = []
    pix_logs_list = []
    for i in tqdm(range(0, num_objs)):
        obj_file_path = f"/Users/sophi/PycharmProjects/MASTERS_PROJECT/map_the_mesh/Assets/textured_mesh/{i}/mesh_update_n.obj"
        ring_logs_s, ring_pix_s, pix_tri_s, pix_logs_s = get_flvae_data_from_obj(obj_file_path,num_one_rings,num_samples)
        # Append data to lists
        ring_logs_list.append(ring_logs_s)
        ring_pix_list.append(ring_pix_s)
        pix_tri_list.append(pix_tri_s)
        pix_logs_list.append(pix_logs_s)

    # Concatenate lists into single JAX arrays
    ring_logs = jnp.stack(ring_logs_list)

    ring_pix = jnp.stack(ring_pix_list)

    pix_tri = jnp.stack(pix_tri_list)

    pix_logs = jnp.stack(pix_logs_list)

    return ring_logs, ring_pix, pix_tri, pix_logs

if __name__ == '__main__':
    # Example usage
    i = 1
    obj_file_path = f"/Users/sophi/PycharmProjects/MASTERS_PROJECT/map_the_mesh/Assets/textured_mesh/{i}/mesh_update_n.obj"
    num_samples = 64
    num_one_rings = 10

    sampler = SampleObj(obj_file_path, num_samples, num_one_rings)
    sampler.sample()
    ring_logs, ring_pix, pix_tri, pix_logs = sampler.get_data()

    print(f"ring_logs: {ring_logs.shape}")
    print(f"{ring_logs[0,0]}")

    print(f"ring_pix: {ring_pix.shape}")
    print(f"{ring_pix[0,0]}")

    print(f"pix_tri: {pix_tri.shape}")
    print(f"{pix_tri[0,0]}")

    print(f"pix_logs: {pix_logs.shape}")
    print(f"{pix_logs[0,0]}")
