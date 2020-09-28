import mido
import numpy as np
from utils.plotting import plot_data


class Note:
    def __init__(self, pitch, on_tick, velocity):
        self.pitch = pitch
        self.on_tick = on_tick
        self.off_tick = -1
        self.velocity = velocity

        self.quantised = False
        self.simplified = False

    def set_off_tick(self, off_tick):
        self.off_tick = off_tick

    def quantise(self, min_ppq):
        self.on_tick = ((self.on_tick + min_ppq / 2) / min_ppq) * min_ppq

    def simplify(self, drum_conversion_dict):
        if self.pitch in drum_conversion_dict.keys():
            self.pitch = drum_conversion_dict[self.pitch]

    def offed(self):
        return not self.off_tick == -1


class NoteList:
    def __init__(self):
        self.notes = []
        self.simplified_notes = []
        self.quantised = False
        self.simplified = False

        '''
        PPQ = 480  # Pulse per quarter note
        event_per_bar = 16
        self.min_ppq = PPQ / (event_per_bar / 4)
        '''

        # ignore: 39 Hand Clap, 54 Tambourine, 56 Cowbell, 58 Vibraslap, 60-81
        self.ignore_pitches = [39, 54, 56, 58]

        self.drum_conversion = {
            35: 36,  # Acoustic Bass Drum (35) -> Bass Drum (36)
            37: 38, 40: 38,  # Side Stick (37), Electric Snare (40) -> Acoustic Snare (38)
            43: 41,  # High Floor Tom (43) -> Low Floor Tom (41)
            47: 45,  # Low-mid Tom (47) -> Low Tom (45)
            50: 48,  # High Tom (50) -> Hi-mid Tom (48)
            44: 42,  # Pedal Hi-Hat (44) -> Closed Hi-Hat (42)
            57: 49, 52: 49,  # Crash 2 (57), China Cymbal (52) -> Crash 1 (49)
            59: 51, 53: 51, 55: 51  # Ride 2 (59), Ride Bell (53), Splash (55) -> Ride 1 (51)
        }

        self.allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]  # Open Hihat (46)
        self.pitch_to_y_dict = {
            36: 0,
            38: 1,
            42: 2,
            46: 3,
            41: 4,
            45: 5,
            48: 6,
            51: 7,
            49: 8
        }
        self.cymbals_pitch = [49, 51]  # Crash, Ride

    def add_note(self, _note):
        self.notes.append(_note)

    def quantise(self):
        if not self.quantised:
            for note in self.notes:
                assert isinstance(note, Note)
                if not note.quantised:
                    note.quantise(min_ppq=self.get_min_ppq())

    def simplify_drums(self):
        if not self.simplified:
            for note in self.notes:
                assert isinstance(note, Note)
                if not note.simplified:
                    note.simplify(drum_conversion_dict=self.drum_conversion)

    def reorganize_by_on_tick(self):
        ordered_notes = {}
        for note in self.notes:
            assert isinstance(note, Note)
            if note.on_tick in ordered_notes.keys():
                ordered_notes[note.on_tick].append(note.pitch)
            else:
                ordered_notes[note.on_tick] = [note.pitch]

        return ordered_notes

    def get_min_ppq(self):
        min_ppq = 9999
        ordered_notes = self.reorganize_by_on_tick()
        on_ticks = list(ordered_notes.keys())
        margin_dict = {}
        for i, on_tick in enumerate(on_ticks[:-1]):
            next_on_tick = on_ticks[i+1]
            margin = next_on_tick - on_tick
            if margin > 0:
                if margin in margin_dict:
                    margin_dict[margin] += 1
                else:
                    margin_dict[margin] = 1
                min_ppq = min(margin, min_ppq)
        return min_ppq, margin_dict[min_ppq]

    def get_last_on_tick(self):
        ordered_notes = self.reorganize_by_on_tick()
        on_ticks = list(ordered_notes.keys())
        return on_ticks[-1]

    def save_to_midi(self, path):
        mid = mido.MidiFile()
        drum_track = mido.MidiTrack()

        min_ppq = self.get_min_ppq()
        ordered_notes = self.reorganize_by_on_tick()
        current_tick = 0

        for on_tick, pitches in ordered_notes.items():
            tick_margin = on_tick - current_tick
            current_tick = max(current_tick, on_tick)

            note_on = mido.Message('note_on', time=tick_margin, note=pitches[0], channel=9)
            drum_track.append(note_on)
            for pitch in pitches[1:]:
                note_on = mido.Message('note_on', time=0, note=pitch, channel=9)
                drum_track.append(note_on)

            tick_margin = min_ppq
            current_tick = current_tick + min_ppq
            note_off = mido.Message('note_off', time=tick_margin, note=pitches[0], channel=9)
            drum_track.append(note_off)
            for pitch in pitches[1:]:
                note_off = mido.Message('note_off', time=0, note=pitch, channel=9)
                drum_track.append(note_off)

        mid.tracks.append(drum_track)
        mid.save(path)

    def get_nonzeros(self):
        min_ppq = self.get_min_ppq()
        self.simplify_drums()
        nonzeros = []
        ordered_notes = self.reorganize_by_on_tick()
        for on_tick, pitches in ordered_notes.items():
            x = int(on_tick / min_ppq)
            for pitch in pitches:
                y = self.pitch_to_y_dict[pitch]

                nonzeros.append([x, y])

        return np.array(nonzeros)


def get_measure_length_and_matrix_nonzeros_from_midi(path):
    mid = mido.MidiFile(path)
    notelist = generate_notelist_from_midi(path)

    ppq = mid.ticks_per_beat  # fourth_note length
    assert ppq % 48 == 0

    min_ppq = ppq // 48  # 1/192 note
    ticks_per_measure = ppq * 4

    notes_num_per_measure = 192
    bars_num = int(notelist.get_last_on_tick() / ticks_per_measure) + 1
    notelist.simplify_drums()
    # drum_matrix = np.zeros((bars_num, notes_num_per_measure, 9))
    nonzeros = []

    ordered_notes = notelist.reorganize_by_on_tick()
    for on_tick, pitches in ordered_notes.items():
        if not on_tick % min_ppq == 0:
            pass
        tick_index = on_tick // min_ppq

        measure_index = tick_index // notes_num_per_measure
        x = tick_index % notes_num_per_measure

        for pitch in pitches:
            if pitch not in notelist.ignore_pitches:
                y = notelist.pitch_to_y_dict[pitch]
                nonzeros.append([measure_index, x, y])

    return bars_num, nonzeros


def generate_notelist_from_midi(path):
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat  # 4th note length
    notelist = NoteList()
    on_notes = []
    current_tick = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if not msg.is_meta and msg.type in ['note_on', 'note_off']:
                # print(msg)
                if msg.type == 'note_on':
                    pitch, tick, velocity = msg.note, msg.time, msg.velocity
                    if pitch == 0:
                        pass
                        # current_tick += tick
                    else:
                        current_tick += tick
                        # print(pitch, current_tick)
                        on_notes.append(Note(pitch=pitch, on_tick=current_tick, velocity=velocity))
                else:  # note_off
                    pitch, tick, velocity = msg.note, msg.time, msg.velocity
                    if pitch == 0:
                        pass
                    current_tick += tick
                    for on_note in on_notes:
                        assert isinstance(on_note, Note)
                        if not on_note.offed() and on_note.pitch == pitch:
                            on_notes.remove(on_note)

                            on_note.set_off_tick(current_tick)
                            notelist.add_note(on_note)
            else:
                pass
                # current_tick += msg.time
    return notelist


def get_metre_list(path):
    mid = mido.MidiFile(path)
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'time_signature':
                print(msg)


def test_get_matrix():
    path = "E:/thrash_drums/Metallica/Master of Puppets/02 - Master of Puppets/drums/1.mid"
    matrix = get_matrix_from_midi(path)
    plot_data(matrix[:1, :, :])
    print(matrix.shape)


if __name__ == '__main__':
    test_get_matrix()