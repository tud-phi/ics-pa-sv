from pathlib import Path
import shutil

TMP_DIR = Path("tmp") / "assignment"


def main():
    # prompt the student for their student id
    student_id = input("Enter your student id (for example 1234567): ")
    # prompt the student for their first name
    first_name = input("Enter your first name (for example Ruben): ")
    # prompt the student for their last name
    last_name = input("Enter your last name (for example MartinRodriguez): ")
    # filename
    archive_filepath = Path(f"{student_id}_{first_name}_{last_name}")

    # delete the temporary folder if it exists
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    # create temporary folder with assignment
    shutil.copytree("assignment", TMP_DIR)

    # delete the datasets of problem 1
    if (TMP_DIR / "problem1" / "datasets").exists():
        shutil.rmtree(TMP_DIR / "problem1" / "datasets")

    shutil.make_archive(archive_filepath, "zip", TMP_DIR)

    print(f"Archive {archive_filepath.resolve()}.zip created successfully")

    # delete the temporary folder
    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()
