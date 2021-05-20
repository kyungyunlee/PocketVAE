import os
import csv
from glob import glob
import random
import numpy as np
import joblib
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf

from data_utils import midi2matrix, train_PCA_per_genre, my_skeletonify, pca_skeletonify, NOTE_DENSITY_CLASSES, VEL_CLASSES, MT_CLASSES
from src.utils import compute_note_density_idx, compute_vel_contour, compute_laidbackness


np.random.seed(0)

@hydra.main(config_path="../src/config", config_name="config")
def gather_files(cfg: DictConfig) -> None:
    """ Collect all data files, per genre.
    Split into train/valid/test n_splits times
    """
    # print(OmegaConf.to_yaml(cfg))
    split_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.data_split_dir)

    # If dataset not splitted yet
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir, exist_ok=True)

    csv_columns = ["track_path"] + \
        [f"split{i}" for i in range(cfg.n_data_splits)]

    genre_list_altnames = {
        "electronic": ["electronic", "Electronic", "Breakbeat", "Dnb", "Downtempo", "Garage", "House", "Jungle", "Old Skool", "Techno", "Trance", "Drum &amp; Bass"],
        "funk": ["funk", "Funk"],
        "rock": ["rock", "Rock"],
        "jazz": ["jazz", "Jazz"],
        "hiphop": ["hiphop", "Hiphop", "Hip Hop", "HIPHOP", "HipHop", "HIP HOP", "trap", "Trap"],
        "blues": ["blues", "Blues"]
    }

    for g, alt_genre_names in genre_list_altnames.items():
        print(g, alt_genre_names)
        all_files = []
        for genre in alt_genre_names:
            # Gmonkee
            gmonkee_files = glob(
                f"{hydra.utils.get_original_cwd()}/{cfg.gmonkee_dir}/**/*{genre}*.mid",
                recursive=True)
            gmonkee_files = [
                i for i in gmonkee_files if "fill" not in i.lower()]
            all_files.extend(gmonkee_files)

            # BFD
            bfd_files = glob(
                f"{hydra.utils.get_original_cwd()}/{cfg.bfd_dir}/**/*{genre}*.mid",
                recursive=True)
            bfd_files = [i for i in bfd_files if "fill" not in i.lower()]
            all_files.extend(bfd_files)

            # GrooveMIDI
            gmidi_files = glob(
                f"{hydra.utils.get_original_cwd()}/{cfg.gmidi_dir}/**/*{genre}*.mid",
                recursive=True)
            gmidi_files = [i for i in gmidi_files if "fill" not in i.lower()]
            all_files.extend(gmidi_files)

            if g == "hiphop":
                # Reddit
                reddit_files = glob(
                    f"{hydra.utils.get_original_cwd()}/{cfg.reddit_dir}/**/*{genre}*.mid",
                    recursive=True)
                reddit_files = [
                    i for i in reddit_files if "fill" not in i.lower()]
                all_files.extend(reddit_files)

        all_files = list(set(all_files))
        print(len(all_files))

        # Open CSV file
        csv_fname = f"{split_dir}/{g}.csv"

        with open(csv_fname, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            # Write to the csv file

            print("before", g, len(all_files))
            valid_files = []
            for fname in all_files:
                out = midi2matrix(fname, cfg.resolution, cfg.bar_length)
                if out:
                    valid_files.append(fname)

            print("after", g, len(valid_files))

            # Split train/valid/test
            list_of_splits = []
            for n in range(cfg.n_data_splits):
                n_train = int(len(valid_files) * 0.8)
                n_test = (len(valid_files) - n_train) // 2
                n_valid = len(valid_files) - n_train - n_test

                split_types = ["train"] * n_train + \
                    ["valid"] * n_valid + ["test"] * n_test

                random.shuffle(split_types)

                list_of_splits.append(split_types)

            print(len(list_of_splits))

            for i, fname in enumerate(valid_files):
                fname = fname.split(hydra.utils.get_original_cwd())[-1][1:]
                data_dict = {"track_path": fname}

                for n in range(cfg.n_data_splits):
                    data_dict[f"split{n}"] = list_of_splits[n][i]

                writer.writerow(data_dict)


@hydra.main(config_path="../src/config", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    preprocessed_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.preprocess_dir)
    split_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.data_split_dir)
    seq_len = cfg.resolution * cfg.bar_length * cfg.beats_per_bar
    hop_len = seq_len // 2

    if not os.path.isdir(preprocessed_dir):
        os.makedirs(preprocessed_dir, exist_ok=True)

    # midi2matrix
    file_list = [[] for _ in range(len(cfg.genre_list))]  # For each genres
    for i, genre in enumerate(cfg.genre_list):
        csv_fname = f"{split_dir}/{genre}.csv"
        fs = []
        split_list = []  # Is it train/valid/test?

        with open(csv_fname, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                splits = []
                fs.append(
                    os.path.join(
                        hydra.utils.get_original_cwd(),
                        row["track_path"]))
                for n in range(cfg.n_data_splits):
                    splits.append(row[f"split{n}"])
                split_list.append(splits)

        process_list = joblib.Parallel(
            n_jobs=-1,
            verbose=1)(
            joblib.delayed(midi2matrix)(
                midi_file,
                resolution=cfg.resolution,
                bar_length=cfg.bar_length) for midi_file in fs)
        process_list = [m + [i] for m in process_list if m is not None]
        assert len(process_list) == len(split_list)

        file_list[i] = [process_list[u] + [split_list[u]]
                        for u in range(len(split_list))]

    # Per split, go through each genre and segment the tracks into 2 or 4 bars
    for n in range(cfg.n_data_splits):
        seg_list = [[] for _ in range(len(cfg.genre_list))]
        for x, genre in enumerate(cfg.genre_list):
            curr_data = file_list[x]
            print(genre, len(curr_data))

            for j, data in enumerate(curr_data):
                note, vel, mt, tempo, midi_f, genre, splits = data
                split = splits[n]  # GET CURRENT SPLIT

                # Repeat if length is too short
                while note.shape[0] < seq_len:
                    note = np.concatenate([note, note], axis=0)
                    vel = np.concatenate([vel, vel], axis=0)
                    mt = np.concatenate([mt, mt], axis=0)

                # Break into seq_len segments
                for i in range(0, note.shape[0], hop_len):
                    if i + seq_len > note.shape[0]:
                        break
                    small_note = note[i: i + seq_len]
                    small_vel = vel[i: i + seq_len]
                    small_mt = mt[i: i + seq_len]

                    # Filter data if there is not enough notes
                    n_kicks = np.sum(small_note[:, 0])
                    n_snares = np.sum(small_note[:, 1])
                    inst_sum = np.sum(small_note, axis=0)
                    nonzero_inst = np.count_nonzero(inst_sum)

                    if n_kicks == 0 and genre != 3:
                        continue
                    elif n_snares == 0 and genre != 3:
                        continue
                    elif nonzero_inst < 3:
                        continue

                    # Compute note density
                    note_density_idx = compute_note_density_idx(
                        small_note, NOTE_DENSITY_CLASSES)

                    # Compute vel contour
                    vel_contour = compute_vel_contour(
                        small_vel, small_note, VEL_CLASSES)

                    # Compute laidbacknesss
                    time_contour = compute_laidbackness(small_mt, small_note)

                    seg_list[x].append(
                        {
                            "note": small_note,
                            "vel": small_vel,
                            "mt": small_mt,
                            "tempo": tempo,
                            "midi_f": midi_f,
                            "genre": genre,
                            "split": split,
                            "note_density_idx": note_density_idx,
                            "vel_contour": vel_contour,
                            "time_contour": time_contour
                        })

        for x in range(len(cfg.genre_list)):
            print(cfg.genre_list[x], len(seg_list[x]))

        # Compute PCA and skeleton and append to data
        note_data = []
        vel_data = []
        for x in range(len(cfg.genre_list)):
            curr_data = seg_list[x]
            notes = [curr_d["note"] for curr_d in curr_data]
            vels = [curr_d["vel"] for curr_d in curr_data]
            note_data.append(notes)
            vel_data.append(vels)

        # Perform PCA
        pca_list, avg_list = train_PCA_per_genre(note_data, cfg.genre_list)
        pca_pickle_fname = os.path.join(preprocessed_dir, f"pca_avg_{n}.pkl")
        pickle.dump([pca_list, avg_list], open(pca_pickle_fname, "wb"))
        print(f"Saved PCA results to {pca_pickle_fname}")

        # Make skeleton
        skeleton_list = [[] for _ in range(len(cfg.genre_list))]

        for x in range(len(cfg.genre_list)):
            curr_note_data = np.array(note_data[x])
            curr_vel_data = np.array(vel_data[x])

            eigendrums = pca_list[x].components_
            avg_matrix = avg_list[x]  # np.mean(np.array(seg_list[x]), axis=0)

            # # Combine hihat+hihatopen+ride+crash
            # curr_note_data_combined = np.zeros_like(curr_note_data)
            # curr_note_data_combined[:,:,0] = curr_note_data[:,:,0]
            # curr_note_data_combined[:,:,1] = curr_note_data[:,:,1]
            # tmp = curr_note_data[:,:,2] + curr_note_data[:,:,3] + curr_note_data[:,:,5] + curr_note_data[:,:,6]
            # tmp[tmp>1] = 1
            # curr_note_data_combined[:,:,2] = tmp

            # pca_skeletons = pca_skeletonify(curr_note_data, avg_matrix, curr_vel_data, eigendrums)
            my_skeletons = []
            for s in range(len(curr_note_data)):
                seg_skel = my_skeletonify(
                    curr_note_data[s][np.newaxis, :], curr_vel_data[s][np.newaxis, :], resolution=cfg.resolution)
                my_skeletons.append(seg_skel)
            # new_skels = np.zeros_like(pca_skeletons)

            # new_skels[:,:,0] = my_skeletons[:,:,0]
            # new_skels[:,:,1] = my_skeletons[:,:,1]
            # new_skels[:,:,2] = pca_skeletons[:,:,2]
            # new_skels[:,:,5] = pca_skeletons[:,:,5]

            # Repeat first bar
            # new_skels[:,16:, :] = new_skels[:,:16,:]
            # my_skeletons[:, 16:, :] = my_skeletons[:,:16,:]

            skeleton_list[x] = np.vstack(np.array(my_skeletons))

        data_list = []
        for x in range(len(cfg.genre_list)):
            data = seg_list[x]
            skeletons = skeleton_list[x]

            for d, s in zip(data, skeletons):
                # note_density = compute_note_density(data["note"])
                # d.update({"note_density": note_density})
                d.update({"skeleton": s})
                data_list.append(d)

        # save to pickle
        pickle_fname = os.path.join(preprocessed_dir, f"all_data_{n}.pkl")
        pickle.dump(data_list, open(pickle_fname, 'wb'))
        print(f"Processed data saved to {pickle_fname}")


if __name__ == "__main__":

    gather_files()
    preprocess()
