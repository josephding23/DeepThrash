import sys
import os
import pdb
# import pretty_midi
import mido

from preprocess.drum_note_preprocess import *


def text_to_notes(encoded_drums, note_list=None):
    if note_list is None:
        note_list = NoteList()

    for word_idx, word in enumerate(encoded_drums):
        c_tick_here = word_idx * min_ppq

        for pitch_idx, pitch in enumerate(allowed_pitch):

            if word[pitch_idx + 2] == '1':
                new_note = Note(pitch, int(c_tick_here))
                # print(pitch, c_tick_here)
                note_list.add_note(new_note)
    return note_list


def conv_text_to_midi(filename):
    f = open(filename, 'r')
    f.readline()
    f.readline()
    sentence = f.readline()
    encoded_drums = sentence.split(' ')

    print(sentence)
    print(encoded_drums)

    # find the first BAR
    first_bar_idx = encoded_drums.index('BAR')
    encoded_drums = encoded_drums[first_bar_idx:]

    try:
        encoded_drums = [ele for ele in encoded_drums if ele not in ['BAR', 'SONG_BEGIN', 'SONG_END', '']]
    except:
        pdb.set_trace()

    # prepare output
    note_list = NoteList()

    mid = mido.MidiFile()
    drum_track = mido.MidiTrack()
    '''
    pm = pretty_midi.PrettyMIDI()
    drum_track = pretty_midi.Instrument(is_drum=True, program=0)
    '''

    PPQ = 220
    min_ppq = PPQ / (event_per_bar/4)
    drum_track.resolution = PPQ # ???? too slow. why??
    # track.resolution = 192
    # pm.instruments.append(drum_track)
    mid.tracks.append(drum_track)

    velocity = 84
    duration = min_ppq * 9 / 10

    note_list = text_to_notes(encoded_drums, note_list=note_list)

    max_c_tick = 0
    not_yet_offed = []
    for note_idx, note in enumerate(note_list.notes[:-1]):
        # add onset
        # tick_here = note.c_tick - max_c_tick
        tick_here = note.c_tick
        pitch_here = note.pitch
        # print(tick_here, pitch_here)
        # pitch_here = pitch_to_midipitch[note.pitch]

        # if pitch_here in cymbals_pitch: # "Lazy-off" for cymbals
        # 	off = midi.NoteOffEvent(tick=0, pitch=pitch_here)
        # 	track.append(off)

        on = mido.Message('note_on', note=pitch_here, velocity=velocity, time=int(tick_here))
        drum_track.append(on)
        max_c_tick = max(max_c_tick, note.c_tick)
        # add offset for something not cymbal

        if note_list.notes[note_idx+1].c_tick == note.c_tick:
            not_yet_offed.append((pitch_here, note.c_tick))

        # else:
        # check out some note that not off-ed.
        start_tick = tick_here
        for off_idx, (waiting_pitch, waiting_tick) in enumerate(not_yet_offed):
            if off_idx == 0:
                off = mido.Message('note_off', note=waiting_pitch, time=int(duration))
                # print(waiting_pitch, waiting_tick, waiting_tick+duration)
                # midi_note = pretty_midi.Note(velocity=velocity, pitch=waiting_pitch,
                #                              start=waiting_tick, end=waiting_tick+duration)
                max_c_tick = max_c_tick + duration
            else:
                off = mido.Message('note_off', note=waiting_pitch, time=int(duration))
                # print(waiting_pitch, waiting_tick, waiting_tick + duration)
                # midi_note = pretty_midi.Note(velocity=velocity, pitch=waiting_pitch,
                #                              start=waiting_tick, end=waiting_tick)

            drum_track.append(off)
            # drum_track.notes.append(midi_note)
            not_yet_offed = []

    if not note_list.notes:
        print(f'No notes in {filename}')
        pdb.set_trace()
    note = note_list.notes[-1]
    tick_here = note.c_tick - max_c_tick
    pitch_here = note.pitch

    on = mido.Message('note_on', time=int(tick_here), velocity=velocity, note=pitch_here)
    off = mido.Message('note_off', time=int(tick_here) + int(duration), note=pitch_here)
    # midi_note = pretty_midi.Note(velocity=velocity, pitch=pitch_here,
    #                              start=tick_here, end=tick_here + duration)

    for off_idx, (waiting_pitch, waiting_tick) in enumerate(not_yet_offed):
        off = mido.Message('note_off', time=waiting_tick, pitch=waiting_pitch)
        # midi_note = pretty_midi.Note(velocity=velocity, pitch=waiting_pitch,
        #                              start=waiting_tick, end=waiting_tick)

    # drum_track.notes.append(midi_note)
    # pm.write(f'{filename[:-4]}.mid')
    new_filename = f'{filename[:-4]}.mid'.replace('models', 'generated_music')
    dirname = os.path.dirname(new_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    mid.save(new_filename)


if __name__ == '__main__':
    result_dir = '../static/models/result_word__256_128_units'
    filenames = os.listdir(result_dir)  # specify which folder result_*.txt files are stored in
    filenames = [f for f in filenames if f.startswith('result') and f.endswith('.txt')]
    filenames = [f for f in filenames if os.path.getsize(result_dir + '/' + f) != 0]
    print(filenames)
    for filename in filenames:
        conv_text_to_midi(result_dir + '/' + filename)

    print(f'Texts -> midi done! for {len(filenames)}')
