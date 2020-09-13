import mido
import numpy as np


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
        self.drum_conversion = {
            35: 36,  # Acoustic Bass Drum (35) -> Bass Drum (36)
            37: 38, 40: 38,  # Side Stick (37), Electric Snare (40) -> Acoustic Snare (38)
            43: 41,  # High Floor Tom (43) -> Low Floor Tom (41)
            47: 45,  # Low-mid Tom (47) -> Low Tom (45)
            50: 48,  # High Tom (50) -> Hi-mid Tom (48)
            44: 42,  # Pedal Hi-Hat (44) -> Closed Hi-Hat (42)
            57: 49, 52: 49,  # Crash 2 (57), China Cymbal (52) -> Crash 1 (49)
            59: 51, 53: 51, 55: 51,  # Ride 2 (59), Ride Bell (53), Splash (55) -> Ride 1 (51)
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
        if _note.off_tick - _note.on_tick == 60:
            print(_note.off_tick, _note.on_tick)

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
        for i, on_tick in enumerate(on_ticks[:-1]):
            next_on_tick = on_ticks[i+1]
            min_ppq = min(next_on_tick - on_tick, min_ppq)
        return min_ppq

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

    def get_matrix(self):
        min_ppq = self.get_min_ppq()
        max_idx = int(self.get_last_on_tick() / self.get_min_ppq()) + 1
        self.simplify_drums()
        drum_matrix = np.zeros((max_idx, 9))

        ordered_notes = self.reorganize_by_on_tick()
        for on_tick, pitches in ordered_notes.items():
            x = int(on_tick / min_ppq)
            for pitch in pitches:
                y = self.pitch_to_y_dict[pitch]

                drum_matrix[x, y] = 1.0

        return drum_matrix

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


def generate_notelist_from_midi_test():
    path = "E:/thrash_drums/Metallica/Kill 'Em All/01 - Hit the Lights/Hit The Lights.mid"
    mid = mido.MidiFile(path)
    notelist = NoteList()
    on_notes = []
    current_tick = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if not msg.is_meta and msg.channel == 9 and msg.type in ['note_on', 'note_off']:
                if msg.type == 'note_on':
                    pitch, tick, velocity = msg.note, msg.time, msg.velocity
                    if pitch == 0:
                        pass
                        # current_tick += tick
                    else:
                        current_tick += tick
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
    print(len(on_notes))
    return notelist


if __name__ == '__main__':
    note_list = generate_notelist_from_midi_test()

    # for note in note_list.notes:
    #     print(note.pitch, note.on_tick, note.off_tick)
    # print(note_list.min_ppq, note_list.last_on_tick)
    # print(note_list.return_as_matrix())
    # note_list.save_to_midi('../static/midi/test/save_test.mid')
    print(note_list.get_nonzeros())