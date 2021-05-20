from torch.utils.data import Dataset
import numpy as np


class DrumDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        curr_data = self.data_list[idx]
        skel = curr_data['skeleton']
        note = curr_data['note']
        vel = curr_data['vel']
        mt = curr_data['mt']
        tempo = curr_data['tempo']
        fname = curr_data['midi_f']
        genre = curr_data['genre']
        note_density_idx = curr_data['note_density_idx']
        vel_contour = curr_data['vel_contour']
        time_contour = curr_data["time_contour"]
        # time_contour =

        # skel = skel[ np.newaxis, :]
        # note = note[np.newaxis, :]
        # vel = vel* 127 // 4

        # range1 = 2
        # range2 = 100

        # mt = (range2 * (mt + (range1 /2))) / range1 - (range2/2) + 50

        # vel = vel / 32
        # mt = mt / 100

        n_inst = np.sum(note, 0)
        n_inst[n_inst > 1] = 1

        n_inst = int(np.sum(n_inst)) - 1

        return {
            "skel": skel,
            "note": note,
            "vel": vel,
            "mt": mt,
            "tempo": tempo,
            "fname": fname,
            "genre": genre,
            "note_density_idx": note_density_idx,
            "vel_contour": vel_contour,
            # "vel_accent": vel_accent,
            "time_contour": time_contour,
            "n_inst": n_inst
            # "time_mode": time_mode
            # "deco": deco
        }
