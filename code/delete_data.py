import os

def delete_database():
    db_file = 'salon_schedule.db'
    if os.path.exists(db_file):
        confirm = input(f"Are you sure you want to delete the database '{db_file}'? This action cannot be undone. (yes/no): ")
        if confirm.lower() == 'yes':
            os.remove(db_file)
            print(f"Database '{db_file}' has been deleted.")
        else:
            print("Deletion canceled.")
    else:
        print(f"Database '{db_file}' does not exist.")

if __name__ == '__main__':
    delete_database()
