import re
from argparse import ArgumentParser
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame, read_csv
from sys import stdout
import psycopg2


def main(args, conn):
    print('==========')
    
    print('Iterating over dataset...\nNotes cleaned:')
    i = 0
    
    clean_df = DataFrame()
    
    cur = conn.cursor()
    cur.execute('''select * from noteevents limit 100;''')
    notes, patients, date, time, admissions = [], [], [], [], []
    
    for row in cur.fetchall():
        patients.append(row[1])
        admissions.append(row[2])
        date.append(row[3])
        time.append(row[4])
        text = re.sub('[^a-zA-Z.]', ' ', row[10].lower())
        words = word_tokenize(re.sub(r'\s+', ' ', text))
        notes.append(' '.join([w for w in words if w not in stopwords.words('english')]))
        i += 1
        stdout.write('\r')
        stdout.flush()
        stdout.write(str(i))
        stdout.flush()

    clean_df['subject_id'] = patients
    clean_df['hadm_id'] = admissions
    clean_df['date'] = date
    clean_df['time'] = time
    clean_df['text'] = notes

    clean_df.to_csv(args.notes_output_name, index=False)


if __name__ == '__main__':

    conn = psycopg2.connect(database="mimic3", user = "ljw", password = "123456", host = "localhost", port = "5432")

    parser = ArgumentParser()
    parser.add_argument('-n', dest='notes_output_name', type=str, default='cleaned-noteevents.csv',
        help='path for output .csv file containing cleaned notes')
    
    main(parser.parse_args(), conn)
    conn.commit()
    conn.close()