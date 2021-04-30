"""sketch of what a VolToSurfMasker could look like."""
import numpy as np
from nilearn import surface, datasets, _utils, plotting


class VolToSurfMasker:
    def fit(self, X=None, y=None):
        fsaverage = datasets.fetch_surf_fsaverage()
        self._meshes = {
            "left": {
                "outer": surface.load_surf_mesh(fsaverage["pial_left"]),
                "inner": surface.load_surf_mesh(fsaverage["white_left"]),
                "bg": surface.load_surf_data(fsaverage["sulc_left"]),
            },
            "right": {
                "outer": surface.load_surf_mesh(fsaverage["pial_right"]),
                "inner": surface.load_surf_mesh(fsaverage["white_right"]),
                "bg": surface.load_surf_data(fsaverage["sulc_right"]),
            },
        }
        start = 0
        for mesh_info in self._meshes.values():
            dim = mesh_info["outer"].coordinates.shape[0]
            mesh_info["dim"] = dim
            mesh_info["slice"] = slice(start, start + dim)
            start += dim
        self._dim = start
        return self

    def transform(self, img, y=None):
        print("load images")
        img = _utils.check_niimg(img, atleast_4d=True)
        out = np.empty((img.shape[-1], self._dim))
        print("project to surface")
        for mesh_name, mesh_info in self._meshes.items():
            print(f"    {mesh_name}")
            out[:, mesh_info["slice"]] = surface.vol_to_surf(
                img,
                mesh_info["outer"],
                inner_mesh=mesh_info["inner"],
                interpolation="nearest",
            ).T
        print("done")
        return out

    def inverse_transform(self, data):
        raise NotImplementedError(
            "mapping back to volume not implemented (yet?)"
        )

    def inverse_transform_to_surf(self, data):
        out = []
        for img_data in np.atleast_2d(data):
            img_info = {}
            for mesh_name, mesh_info in self._meshes.items():
                img_info[mesh_name] = surface.Surface(
                    mesh_info["outer"], img_data[mesh_info["slice"]]
                )
            out.append(img_info)
        return out

    def bg_map(self, mesh_name):
        return self._meshes[mesh_name]["bg"]


if __name__ == "__main__":
    # example usage adapted from the mixed gambles decoding example

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectPercentile, f_classif

    masker = VolToSurfMasker()
    feature_selection = SelectPercentile(f_classif, percentile=5)
    pipe = Pipeline(
        [
            ("masker", masker),
            ("anova", feature_selection),
            ("svc", LinearSVC()),
        ]
    )

    gambles = datasets.fetch_mixed_gambles(n_subjects=16)

    pipe.fit(gambles.zmaps, gambles.gain)
    coef = pipe.named_steps["anova"].inverse_transform(
        pipe.named_steps["svc"].coef_
    )
    surfaces = masker.inverse_transform_to_surf(coef)[0]

    plotting.view_surf(
        surfaces["left"].mesh,
        surfaces["left"].data,
        threshold=0.05,
        bg_map=masker.bg_map("left"),
    ).open_in_browser()
