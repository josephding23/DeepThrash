import mido
from process.notelist import *
import os
from process.database import *


def scan_local():
    performers_collection = get_performers_table()
    albums_collection = get_albums_table()
    songs_collection = get_songs_table()

    root_dir = 'E:/thrash_drums'
    for performer in os.listdir(root_dir):
        performer_dir = root_dir + '/' + performer
        albums_list = []

        for album in os.listdir(performer_dir):
            albums_list.append(album)

            album_dir = performer_dir + '/' + album
            songs_list = []

            for song in os.listdir(album_dir):

                song_dir = album_dir + '/' + song
                track_no, song_name = song.split(' - ')
                track_no = int(track_no)
                songs_list.append({'TrackNum': track_no, 'Name': song_name})

                midi_file_path = ''
                for file in os.listdir(song_dir):
                    if file[-4:] == '.mid':
                        midi_file_path = song_dir + '/' + file

                assert midi_file_path is not ''

                song_info = {
                    'TrackNum': track_no,
                    'Name': song,
                    'Album': album,
                    'Performer': performer,
                    'MidiPath': midi_file_path
                }
                if songs_collection.count({'Performer': performer, 'Album': album, 'Name': song}) == 0:
                    songs_collection.insert_one(song_info)

            album_info = {
                'Name': album,
                'Performer': performer,
                'Directory': album_dir,
                'SongsList': songs_list
            }
            if albums_collection.count({'Performer': performer, 'Album': album}) == 0:
                albums_collection.insert_one(album_info)

        performer_info = {
            'Name': performer,
            'Directory': performer_dir,
            'AlbumsList': albums_list
        }
        if performers_collection.count({'Name': performer}) == 0:
            performers_collection.insert_one(performer_info)


def analyze_songs_info():
    songs_collection = get_songs_table()
    for song_info in songs_collection.find({'HasDrum': True}):
        path = song_info['MidiPath']
        drum_path = song_info['DrumsDir']
        if os.path.exists(drum_path):
            files_num = len(os.listdir(drum_path))
            for i in range(files_num):
                path = drum_path + f'{i+1}.mid'
                print(path)
                mid = mido.MidiFile(path)
                notelist = generate_notelist_from_midi(path)
                ppq = mid.ticks_per_beat
                min_ppq_info = notelist.get_min_ppq()
                print(notelist.get_min_ppq(), ppq, ppq // min_ppq_info[0])


def generate_all_drum_nonzeros():
    nonzeros_dir = 'E:/thrash_drums/nonzeros/'
    songs_collection = get_songs_table()
    current_measure_index = 0
    all_nonzeros = []
    for song_info in songs_collection.find():
        drum_path = song_info['DrumsDir']
        if os.path.exists(drum_path):
            files_num = len(os.listdir(drum_path))
            for i in range(files_num):
                path = drum_path + f'{i + 1}.mid'
                measures_num, song_nonzeros = get_measure_length_and_matrix_nonzeros_from_midi(path)
                for nonzero in song_nonzeros:
                    # print([nonzero[0] + current_measure_index, nonzero[1], nonzero[2]])
                    all_nonzeros.append([nonzero[0] + current_measure_index, nonzero[1], nonzero[2]])
                current_measure_index += measures_num
    all_nonzeros = np.array(all_nonzeros)
    np.savez(nonzeros_dir + 'all_nonzeros.npz', all_nonzeros)


def generate_bands_drum_nonzeros():
    nonzeros_dir = 'E:/thrash_drums/nonzeros/'
    songs_collection = get_songs_table()
    performers_collection = get_performers_table()

    for performer_info in performers_collection.find():
        band_nonzeros = []
        current_measure_index = 0

        performer = performer_info['Name']
        for song_info in songs_collection.find({'Performer': performer}):
            drum_path = song_info['DrumsDir']
            if os.path.exists(drum_path):
                files_num = len(os.listdir(drum_path))
                for i in range(files_num):
                    path = drum_path + f'{i + 1}.mid'
                    measures_num, song_nonzeros = get_measure_length_and_matrix_nonzeros_from_midi(path)
                    for nonzero in song_nonzeros:
                        # print([nonzero[0] + current_measure_index, nonzero[1], nonzero[2]])
                        band_nonzeros.append([nonzero[0] + current_measure_index, nonzero[1], nonzero[2]])
                    current_measure_index += measures_num
        all_nonzeros = np.array(band_nonzeros)
        np.savez(nonzeros_dir + f'{performer}_nonzeros.npz', all_nonzeros)
        print(f'{performer} nonzeros saved, {len(all_nonzeros)} in total.')


if __name__ == '__main__':
    generate_bands_drum_nonzeros()