import pdb

PPQ = 480  # Pulse per quarter note
event_per_bar = 16
min_ppq = PPQ / (event_per_bar / 4)
# ignore: 39 Hand Clap, 54 Tambourine, 56 Cowbell, 58 Vibraslap, 60-81

drum_conversion = {
    35: 36,  # Acoustic Bass Drum (35) -> Bass Drum (36)
    37: 38, 40: 38,  # Side Stick (37), Electric Snare (40) -> Acoustic Snare (38)
    43: 41,  # High Floor Tom (43) -> Low Floor Tom (41)
    47: 45,  # Low-mid Tom (47) -> Low Tom (45)
    50: 48,  # High Tom (50) -> Hi-mid Tom (48)
    44: 42,  # Pedal Hi-Hat (44) -> Closed Hi-Hat (42)
    57: 49, 52: 49,  # Crash 2 (57), China Cymbal (52) -> Crash 1 (49)
    59: 51, 53: 51, 55: 51,  # Ride 2 (59), Ride Bell (53), Splash (55) -> Ride 1 (51)
}

allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]  # Open Hihat (46)
cymbals_pitch = [49, 51]  # Crash, Ride


class Note:
    def __init__(self, pitch, c_tick):
        self.pitch = pitch
        self.c_tick = c_tick  # cumulated_tick of a midi note
        self.idx = None

    def add_index(self, idx):
        self.idx = idx


class NoteList():
    def __init__(self):
        self.notes = []
        self.quantised = False
        self.max_idx = None

    def add_note(self, note):
        self.notes.append(note)

    def quantise(self, minimum_ppq):
        note = None
        if not self.quantised:
            for note in self.notes:
                note.c_tick = ((note.c_tick + minimum_ppq / 2) / minimum_ppq) * minimum_ppq  # quantise
                note.add_index(note.c_tick / minimum_ppq)

            self.max_idx = note.idx
            if (self.max_idx + 1) % event_per_bar != 0:
                self.max_idx += event_per_bar - (
                            (self.max_idx + 1) % event_per_bar)  # make sure it has a FULL bar at the end.
            self.quantised = True

    def simplify_drums(self):
        for note in self.notes:
            if note.pitch in drum_conversion:  # ignore those not included in the key
                note.pitch = drum_conversion[note.pitch]

        self.notes = [note for note in self.notes if note.pitch in allowed_pitch]

    def return_as_text(self):
        length = self.max_idx + 1  # of events in the track.
        event_track = []
        for note_idx in range(length):
            event_track.append(['0'] * len(allowed_pitch))

        num_bars = length / event_per_bar  # + ceil(len(event_texts_temp) % _event_per_bar)

        for note in self.notes:
            pitch_here = note.pitch
            note_add_pitch_index = allowed_pitch.index(pitch_here)  # 0-8
            event_track[note.idx][note_add_pitch_index] = '1'
        # print note.idx, note.c_tick, note_add_pitch_index, ''.join(event_track[note.idx])
        # pdb.set_trace()

        event_text_temp = ['0b' + ''.join(e) for e in event_track]  # encoding to binary

        event_text = []
        # event_text.append('SONG_BEGIN')
        # event_text.append('BAR')
        for bar_idx in range(num_bars):
            event_from = bar_idx * event_per_bar
            event_to = event_from + event_per_bar
            event_text = event_text + event_text_temp[event_from:event_to]
            event_text.append('BAR')

        # event_text.append('SONG_END')

        return ' '.join(event_text)