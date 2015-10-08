import os
import csv

CSV_folder_path = ''
MVSO_destination_folder = ''


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print 'Creating dir: ', directory


for file_name in os.listdir(CSV_folder_path):
    # Check if the file is a CSV file
    if not file_name.__endswith__('.csv'):
        continue

    # Create the language folder
    language = file_name.split('.')[0]
    language_dir = os.path.join(MVSO_destination_folder, language, '')
    create_dir(language_dir)

    # Open file
    csv_file = open(os.path.join(CSV_folder_path, file_name))

    # Iterate and download images
    first_line = True # first line does not contain any link
    for row in csv_file:
        if first_line:
            first_line = False
            continue
        anp, url = row.split(',')
        anp_dir = os.path.join(language_dir, anp, '')
        create_dir(anp_dir)
        file_path = os.path.join(anp_dir, url.split('/')[-1])
        # Download image (urllib2???)