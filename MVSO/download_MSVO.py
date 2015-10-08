import os
import sys
import csv
import urllib2
import time

CSV_folder_path = ''
MVSO_destination_folder = ''


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print 'Creating dir: ', directory


''' MAIN CODE '''
if len(sys.argv) >= 3:
    try:
        CSV_folder_path = str(sys.argv[1])
        MVSO_destination_folder = str(sys.argv[2])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit('Arguments: CSV_folder_path MVSO_destination_folder')

create_dir(MVSO_destination_folder)

for file_name in os.listdir(CSV_folder_path):
    # Check if the file is a CSV file
    if not file_name.__endswith__('.csv'):
        continue

    print '\n\nDownloading images in ' + file_name + '\n'

    # Create the language folder
    language = file_name.split('.')[0]
    language_dir = os.path.join(MVSO_destination_folder, language, '')
    create_dir(language_dir)

    # Open file
    csv_file = open(os.path.join(CSV_folder_path, file_name))

    # Iterate and download images
    index = 0  # first line does not contain any link
    failed_images = 0
    for row in csv_file:
        if index == 0:
            index += 1
            continue
        anp, url = row.split(',')
        anp_dir = os.path.join(language_dir, anp, '')
        create_dir(anp_dir)
        file_path = os.path.join(anp_dir, url.split('/')[-1])
        # Download image
        try:
            f = urllib2.urlopen(url)
            with open(file_path, "wb") as downloaded_file:
                downloaded_file.write(f.read())
            # Random wait to avoid Flickr banning us
            time.sleep(0.2)
            index += 1
        except:
            failed_images += 1
        if index%20 == 0:
            print '[' + language + '] Images processed: ' + str(index)

    # Close file
    csv_file.close()