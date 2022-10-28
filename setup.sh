
echo "Creating a virtual environment"
python3 -m venv venv

echo "Activating virtual environment"
source venv/bin/activate

echo "Installing required python packages"
python3 -m pip install -r requirements.txt

echo "Adding to the PYTHONPATH"
work_dir="$PWD/"
if [[ "${PYTHONPATH}" =~ "$work_dir" ]];
then
  echo "Already added"
else
  export PYTHONPATH="${PYTHONPATH}:$work_dir"
fi