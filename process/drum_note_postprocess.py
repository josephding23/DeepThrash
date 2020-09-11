import os
# import pretty_midi
import mido
import shutil

from process.notelist import *


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


def get_note_list_from_file(path):
    f = open(path, 'r')
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
    note_list = text_to_notes(encoded_drums, note_list=note_list)
    return note_list


def conv_text_to_midi(path):

    note_list = get_note_list_from_file(path)
    reorganized_notes = note_list.reorganize_by_tick()

    mid = mido.MidiFile()
    drum_track = mido.MidiTrack()
    drum_track.resolution = 192

    PPQ = 220
    min_ppq = PPQ / (event_per_bar / 4)
    drum_track.resolution = PPQ # ???? too slow. why??
    # track.resolution = 192
    # pm.instruments.append(drum_track)
    mid.tracks.append(drum_track)

    velocity = 84
    # duration = min_ppq * 9 / 10
    duration = min_ppq

    max_c_tick = 0

    for c_tick, pitch_list in reorganized_notes.items():
        on_tick = c_tick - max_c_tick
        off_tick = duration

        for c_pitch in pitch_list:
            on = mido.Message('note_on', time=int(on_tick), velocity=velocity, note=c_pitch, channel=9)
            off = mido.Message('note_off', time=int(off_tick), velocity=velocity, note=c_pitch, channel=9)

            drum_track.append(on)
            drum_track.append(off)

        max_c_tick = max(max_c_tick, c_tick + duration)

    new_filename = f'{path[:-4]}.mid'.replace('models', 'generated_music')
    dirname = os.path.dirname(new_filename)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    mid.save(new_filename)


if __name__ == '__main__':
    result_dir = '../static/models/result_word__256_128_units'

    generated_dir = result_dir.replace('models', 'generated_music')
    shutil.rmtree(generated_dir)
    os.mkdir(generated_dir)

    filenames = os.listdir(result_dir)  # specify which folder result_*.txt files are stored in
    filenames = [f for f in filenames if f.startswith('result') and f.endswith('.txt')]
    filenames = [f for f in filenames if os.path.getsize(result_dir + '/' + f) != 0]
    print(filenames)
    for filename in filenames:
        conv_text_to_midi(result_dir + '/' + filename)

    print(f'Texts -> midi done! for {len(filenames)}')
