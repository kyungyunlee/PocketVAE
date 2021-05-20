import numpy as np
import math
import pickle
from collections import Counter
import miditoolkit.midi as mtk_midi
import miditoolkit.midi.containers as mtk_container
from sklearn.decomposition import PCA

# import omegaconf

# cfg = omegaconf.OmegaConf.load("../src/config/config.yaml")

DRUM_CLASSES = [
    "kick",
    "snare",
    "hihat_closed",
    "hihat_open",
    "tom",
    "ride",
    "crash"
]


DRUM2MIDI = {
    "kick": [35, 36],
    "snare": [40, 37, 39, 38, 10001],
    "hihat_closed": [22, 44, 42],
    "hihat_open": [26, 46],
    "tom": [41, 43, 45, 47, 48, 50, 58],
    "ride": [53, 59, 56, 51],
    "crash": [52, 57, 49, 55],
}

IDX2MIDI = {
    0: 35,  # kick
    1: 38,  # snare
    2: 42,  # hh closed
    3: 46,  # hh open
    4: 47,  # tom
    5: 51,  # ride
    6: 49,  # crash
}

DRUM2IDX = {drum: i for i, drum in enumerate(DRUM_CLASSES)}

MIDI2IDX = {
    midi_num: DRUM2IDX[drum] for drum, midi_nums in DRUM2MIDI.items() for midi_num in midi_nums
}

NOTE_DENSITY_CLASSES = [0.075, 0.1, 0.125, 0.150, 0.175, 0.2, 0.225]
VEL_CLASSES = np.arange(0.2, 1.2, 0.2)
MT_CLASSES = [-1, 0, 1]


def load_data(pickle_fname):
    data = pickle.load(open(pickle_fname, 'rb'))
    train_data = []
    valid_data = []
    test_data = []
    for d in data:
        split = d['split']
        # d.pop('split', None)
        if split == "train":
            train_data.append(d)
        elif split == "valid":
            valid_data.append(d)
        elif split == "test":
            test_data.append(d)

    return train_data, valid_data, test_data


def midi2matrix(midi_file, resolution, bar_length):
    print(midi_file)
    # load midi file
    try:
        midi_obj = mtk_midi.parser.MidiFile(midi_file)
    except BaseException:
        print("Failed to load midi: ", midi_file)
        return None

    # get time signature
    ts = midi_obj.time_signature_changes

    if len(ts) == 0:
        time_sig = (4, 4)
    else:
        ts = ts[0]
        time_sig = (ts.numerator, ts.denominator)
    if time_sig != (4, 4):
        print(f"Not 4/4 {midi_file}, {time_sig}")
        return None

    # get tempo
    tempo_changes = midi_obj.tempo_changes
    # print (tempo_changes)
    if len(tempo_changes) == 0:
        tempo = 120
    else:
        tempo_list = [a.tempo for a in tempo_changes]
        tempo = max(tempo_list)

    tempo = int(tempo)
    # print ("tempo",tempo)

    # If there are no instruments
    if len(midi_obj.instruments) == 0:
        print(f"There are no instruments in the midi file {midi_file}")
        return None

    if len(midi_obj.instruments) > 1:
        inst = None
        for inst_ in midi_obj.instruments:
            if inst_.is_drum:
                inst = inst_
        if not inst:
            print(f"There is no drum in the midi file {midi_file}")
            return None
    else:
        inst = midi_obj.instruments[0]

    # 1 beat refers to quarter note
    ticks_per_beat = midi_obj.ticks_per_beat
    total_ticks = max([note.end for note in inst.notes[-10:]])
    total_beats = math.ceil(total_ticks / ticks_per_beat)  # round up

    beats_per_bar = time_sig[1]

    total_bars = math.ceil(total_beats / beats_per_bar)

    if total_bars == 0:
        total_bars = 1

    subdivisions_per_bar = resolution * beats_per_bar
    T = total_beats * resolution
    N = len(DRUM_CLASSES)

    NOTES = np.zeros((T, N))
    VELS = np.zeros((T, N))
    MICROTIMES = np.zeros((T, N))

    ticks_per_subdivision = ticks_per_beat // resolution

    drum_notes = inst.notes
    for i, note in enumerate(drum_notes):
        if note.pitch in MIDI2IDX.keys():
            n_idx = MIDI2IDX[note.pitch]
        else:
            print(note.pitch, "note skip")
            continue

        t_idx = round(note.start / ticks_per_subdivision)

        if t_idx >= T:
            continue

        microtime = note.start - (ticks_per_subdivision * t_idx)

        # range1 = ticks_per_subdivision * 2
        # range2 = 32

        # # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        # microtime = (((microtime + ticks_per_subdivision) * range2) / range1) - range2/2

        # microtime = int (microtime)
        # microtime = float(microtime)
        # microtime = microtime / (range2/2)  # -1 ~ 1 range

        range1 = ticks_per_subdivision
        range2 = 2
        microtime = (range2 * (microtime + ticks_per_subdivision // 2)
                     ) / range1 - (range2 / 2)

        note_duration = note.end - note.start

        if note_duration < ticks_per_subdivision // 4:
            is_roll = True

        else:
            is_roll = False

        vel = note.velocity
        # vel = vel //4 # quantize
        # vel = vel / 32 # 0-1 range
        vel = vel / 127

        if NOTES[t_idx, n_idx] != 1:
            # check for snare roll
            if n_idx == 1 and is_roll:
                n_idx = MIDI2IDX[10001]

                if NOTES[t_idx, n_idx] == 1:
                    continue

            NOTES[t_idx, n_idx] = 1
            VELS[t_idx, n_idx] = vel
            MICROTIMES[t_idx, n_idx] = microtime

    # Check for nvalid files
    # Too little notes
    if np.count_nonzero(NOTES) < 10:
        print(f"Too small notes {midi_file}", NOTES)
        return None

    n_kicks = np.sum(NOTES[:, 0])
    n_snares = np.sum(NOTES[:, 1])
    instrument_sum = np.sum(NOTES, axis=0)
    nonzero_inst = np.count_nonzero(instrument_sum)

    # If there are no kicks
    if n_kicks == 0 and "jazz" not in midi_file.lower():
        return None

    # No snares
    if n_snares == 0 and "jazz" not in midi_file.lower():
        return None

    # If there is only one instrument
    if nonzero_inst < 3:
        return None

    return [NOTES, VELS, MICROTIMES, tempo, midi_file]


def matrix2midi(note_matrix, vel_matrix, microtime_matrix,
                tempo, filename, resolution, only_note=False):

    midi_obj = mtk_midi.parser.MidiFile()
    track = mtk_container.Instrument(program=0, is_drum=True, name="drum set")
    midi_obj.instruments = [track]

    ticks_per_beat = midi_obj.ticks_per_beat = 480

    ticks_per_subdivision = ticks_per_beat // resolution

    if only_note:
        vel_matrix = np.ones_like(note_matrix)
        microtime_matrix = np.zeros_like(note_matrix)

    for i in range(note_matrix.shape[0]):
        idxs = np.where(note_matrix[i] >= 0.5)[0]
        for inst_idx in idxs:

            pitch = IDX2MIDI[inst_idx]

            # mt = microtime_matrix[i, inst_idx]

            # range1 = 2
            # range2 = ticks_per_subdivision * 2
            # # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
            # mt = (((mt +1) * range2) / range1) + (-ticks_per_subdivision)

            # velocity = int (vel_matrix[i, inst_idx] * 127)

            # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

            range1 = 2
            range2 = ticks_per_subdivision
            mt = microtime_matrix[i, inst_idx]
            mt = (range2 * (mt + 1)) / range1 - ticks_per_subdivision // 2

            velocity = int(vel_matrix[i, inst_idx] * 127)

            start = int(i * ticks_per_subdivision + mt)
            end = start + ticks_per_subdivision

            if start < 0:
                start = 0

            # snare roll
            if pitch == 10001:
                pitch = IDX2MIDI[1]
                roll_duration = ticks_per_subdivision // 4
                note1 = mtk_container.Note(
                    start=start,
                    end=start + roll_duration,
                    pitch=pitch,
                    velocity=velocity
                )
                note2 = mtk_container.Note(
                    start=start + roll_duration,
                    end=start + roll_duration * 2,
                    pitch=pitch,
                    velocity=velocity
                )
                midi_obj.instruments[0].notes.append(note1)
                midi_obj.instruments[0].notes.append(note2)
            else:
                note = mtk_container.Note(
                    start=start, end=end, pitch=pitch, velocity=velocity
                )
                midi_obj.instruments[0].notes.append(note)

    midi_obj.tempo_changes = [mtk_container.TempoChange(tempo, 0)]
    midi_obj.dump(filename)


def my_skeletonify(notes, vels, resolution):
    # notes = shape(n_segments, seq_len, n_drum_classes)
    SKELS = np.zeros_like(notes)
    T = notes.shape[1]

    # Remove ghost notes
    # snare_idx = DRUM2IDX["snare"]
    snare_idx = 1
    # min_snare_vel = sorted(vels[:, :, snare_idx][vels[:, :,snare_idx]>0])[::-1]

    if np.count_nonzero(notes[:, :, snare_idx]) > 0:

        # min_snare_vel = min_snare_vel / 127
        # min_snare_vel = 90 / 127
        # snare_neg_idx = np.where(vels[:, :, snare_idx] < min_snare_vel)

        # SKELS[:,:, snare_idx] = notes[:,:,snare_idx]
        # SKELS[snare_neg_idx] = 0
        # 2, 4 beat
        two_four_note_cnt = np.sum(notes[:, :, snare_idx][:, 4::8])
        rest_note_cnt = np.sum(notes[:, :, snare_idx]) - two_four_note_cnt

        two_four_vel = np.sum(vels[:, :, snare_idx][:, 4::8])
        rest_vel = np.sum(vels[:, :, snare_idx]) - two_four_vel

        if two_four_note_cnt > 0:
            two_four_vel = two_four_vel / two_four_note_cnt
        if rest_note_cnt > 0:
            rest_vel = rest_vel / rest_note_cnt
        else:
            rest_vel = 0

        if two_four_note_cnt > T // 8 - 2:
            min_snare_vel = np.min(vels[:, :, snare_idx][:, 4::8])

        else:
            min_snare_vel = rest_vel

        snare_tmp = vels[:, :, snare_idx]
        snare_tmp[snare_tmp < min_snare_vel] = 0
        snare_tmp[snare_tmp > 0] = 1
        SKELS[:, :, snare_idx] = snare_tmp

    # Hihat
    ride_idx = 5
    crash_idx = 6
    hho_idx = 3
    hhc_idx = 2

    note_ride = notes[:, :, ride_idx] + notes[:, :, crash_idx]
    note_hh = notes[:, :, hho_idx] + notes[:, :, hhc_idx]
    # + notes[:,:,crash_idx]

    # Ride or hihat as main
    n_rides = np.sum(note_ride)
    n_hihats = np.sum(note_hh)

    if n_rides > n_hihats:
        n_hats = n_rides
        hat_idx = ride_idx
    else:
        n_hats = n_hihats
        hat_idx = hhc_idx

    # Check if 16th or 8th
    if n_hats >= int(notes.shape[1] * 3 / 4):
        # 16th
        skel_hat = np.ones_like(notes[:, :, hat_idx])
        SKELS[:, :, hat_idx] = skel_hat
    else:
        # 8th
        skel_hat = np.zeros_like(notes[:, :, hat_idx])
        skel_hat[:, ::resolution // 2] = 1
        SKELS[:, :, hat_idx] = skel_hat

    """
    # Ride notes only at 1, &
    SKELS[:, :, ride_idx] = note_ride
    tmp_ride = np.zeros((SKELS.shape[0], SKELS.shape[1],))
    tmp_ride[:, ::resolution // 2] = 1
    SKELS[:, :, ride_idx] = np.multiply(tmp_ride, SKELS[:, :, ride_idx])

    # Hihat
    note_hh[note_hh > 1] = 1
    SKELS[:, :, hhc_idx] = note_hh
    tmp_hh = np.zeros((SKELS.shape[0], SKELS.shape[1],))
    tmp_hh[:, ::resolution // 2] = 1
    SKELS[:, :, hhc_idx] = np.multiply(tmp_hh, SKELS[:, :, hhc_idx])
    """

    # Kick
    kick_idx = 0
    if np.count_nonzero(notes[:, :, kick_idx]) > 0:

        min_kick_vel = 40 / 127
        kick_tmp = vels[:, :, kick_idx]
        kick_tmp[kick_tmp < min_kick_vel] = 0
        kick_tmp[kick_tmp > 0] = 1

        SKELS[:, :, kick_idx] = kick_tmp

    # SKELS[:, :, kick_idx] = notes[:, :, kick_idx]

    # Tom
    tom_idx = 4
    tmp = np.zeros_like(notes[:, :, tom_idx])
    tmp[:, ::resolution // 2] = 1
    tmp = np.multiply(notes[:, :, tom_idx], tmp)
    hat_tmp = SKELS[:, :, hat_idx] + tmp
    hat_tmp[hat_tmp > 1] = 1
    SKELS[:, :, hat_idx] = hat_tmp

    return SKELS


def train_PCA_per_genre(data_list, genre_list):
    """
    data_list : list that contains data per genre
    data_list[0] is data for genre number 0
    """
    pca_list = []
    avg_list = []

    component_list = [3, 4, 4, 5, 3, 1]
    for x in range(len(genre_list)):
        # pca = PCA(n_components, svd_solver='randomized')
        pca = PCA(component_list[x], svd_solver='randomized')
        curr_data = np.array(data_list[x])
        # combine hihat open and hihat closed
        hat = curr_data[:, :, 2] + curr_data[:, :, 3]
        hat[hat > 1] = 1
        curr_data[:, :, 3] = 0
        curr_data[:, :, 2] = hat

        curr_data = curr_data.reshape(len(curr_data), -1)
        avg = np.mean(curr_data, axis=0)
        avg_list.append(avg)
        curr_data = curr_data - avg

        pca = pca.fit(curr_data)
        evr = pca.explained_variance_ratio_
        print(genre_list[x])
        print(evr)
        print(np.sum(evr))
        pca_list.append(pca)
    # print (len(avg_list) )

    return pca_list, avg_list


def pca_skeletonify(matrix, avg_matrix, vel_matrix, eigendrums):
    """
    matrix : original data of shape=(1, seq_len, n_drum_classes)
    """
    # eigendrums = pca_list[genre_id].components_

    # standardize
    len_data = matrix.shape[0]
    seq_len = matrix.shape[1]
    n_class = matrix.shape[2]
    avg_matrix = avg_matrix.reshape(1, -1)
    matrix_reshaped = matrix.reshape(len_data, -1)
    matrix_reshaped = matrix_reshaped - avg_matrix

    weight = np.dot(matrix_reshaped, eigendrums.T)
    # linear combination of eigendrums
    skeleton = avg_matrix + np.dot(weight, eigendrums)
    skeleton = skeleton.reshape(len_data, seq_len, n_class)
    skel_tmp = np.zeros_like(skeleton)
    skel_tmp[skeleton >= 0.5] = 1
    skel_tmp[skeleton < 0.5] = 0

    # KEEP THE KICKS
    # skel_tmp [:, :, 0] =  matrix[:,:,0]

    for i in range(len(skel_tmp)):
        # KEEP MAJOR SNARE
        if np.sum(matrix[i, :, 1]) == 0:
            vel_dominant = vel_matrix[i, :, 1]
            vel_dominant_idx = np.where(vel_dominant >= 0.8)
            tmp = np.zeros_like(vel_dominant)
            tmp[vel_dominant_idx] = 1
            skel_tmp[i, :, 1] = tmp

        if np.sum(skel_tmp[i, :, 1]) > 8:
            tmp_mask = np.zeros_like(matrix[i, :, 1])
            tmp_mask[::4] = 1
            skel_tmp[i, :, 1] = skel_tmp[i, :, 1] * tmp_mask

        # Ride
        if np.sum(matrix[i, :, 5]) > np.sum(matrix[i, :, 2]):
            tmp = matrix[i, :, 5]
            tmp_mask = np.zeros_like(tmp)
            tmp_mask[::2] = 1
            tmp = tmp * tmp_mask
            skel_tmp[i, :, 5] = tmp

        if np.sum(matrix[i, :, 6]) > np.sum(matrix[i, :, 2]):
            tmp = matrix[i, :, 6]
            tmp_mask = np.zeros_like(tmp)
            tmp_mask[::2] = 1
            tmp = tmp * tmp_mask
            skel_tmp[i, :, 6] = tmp

        if np.sum(skel_tmp) == 0:
            skel_tmp = matrix

    return skel_tmp
