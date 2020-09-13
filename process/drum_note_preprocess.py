import mido
from process.notelist import Note, NoteList


def read_drum_test():
    path = "E:/thrash_drums/Metallica/Kill 'Em All/01 - Hit the Lights/Hit The Lights.mid"
    mid = mido.MidiFile(path)

    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            if not msg.is_meta and msg.channel == 9 and msg.type in ['note_on', 'note_off']:
                print(msg)


if __name__ == '__main__':
    read_drum_test()