import regex
import pandas as pd


if __name__ == '__main__':
    with open('labels.txt') as fp:
        data = fp.read()
    moves = []
    append_move = moves.append
    for match in regex.finditer(r'(p(\d+)_c(\d+)_s(\d+)) (?:(\d+)-(\d+):(\d+)\s*)+\n?', data):
        video = match[1]
        person = match[2]
        scenery = match[3]
        for start, end, sentence in zip(match.captures(5), match.captures(6), match.captures(7)):
            append_move({
                'video': video,
                'person': person,
                'scenery': scenery,
                'start': start,
                'end': end,
                'sentence': sentence,
            })
    df = pd.DataFrame(moves)
    df.to_csv('labels.csv', index=False)
    df.to_excel('labels.xlsx', 'labels', engine='xlsxwriter', index=False)
