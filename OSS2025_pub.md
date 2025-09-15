Open Suturing Skills Challenge 2025 ✨
Efficient and precise surgical skills are essential in ensuring positive patient outcomes. Whereas machine learning-based surgical skill assessment is gaining traction for minimally invasive techniques, this cannot be said for open surgery skills. Open surgery generally has more degrees of freedom when compared to minimally invasive surgery, making it more difficult to interpret. By continuously providing real-time, data driven, and objective evaluation of surgical performance, automated skill assessment has the potential to greatly improve surgical skill training.

In this challenge, the aim is to classify surgical skill. We present a challenge dataset containing videos of surgical suturing in a simulated environment. Each video is rated according to the OSATS categories, as well as with a Global Rating Score (GRS). Participants are challenged to complete two tasks for a given video: 1) predict the total GRS score and 2) predeict the full OSATS scoring table. By providing an automated suturing skill feedback, surgeons will be able to easily practice and refine a fundamental surgical skill. This in turn will ameliorate surgical performance, ultimately greatly improving patient outcome.

This year there is also a third task challenging participants in their tool and hand tracking skills. Objective of this task is to effectively track hand and tool keypoints across movement sequences.
(We actively encourage participants to use trained models from task 3 to solve tasks 1 & 2)

For additional information regarding the challenge design, click here (starting at page 37).

📅 Timeline
✅ May 1st, 2025: Website launch and registration opening

✅ June 13th, 2025: Release training data

✅ July 2nd, 2025: Release validation data

✅ July 5th, 2025: Evaluation Scripts

⏳ August 1st, 2025: Start of evaluation

‼️ September 1st 15th, 2025, 11:59 PM GMT: Submission deadline

‼️ September 17th, 2025, 11:59 PM GMT: Write-up and presentation submission deadline

🏆 Semptember 27th, 2025: Challenge Day

Location: DCC 1
Room: DCC1-2F-209-210-211
Time: 11:45 AM (GMT+9)

👷‍♀️ Organizers
NCT Dresden



Hanna Hoffmann	Sebastian Bodenstedt	Stefanie Speidel
University Hospital Essen

Jan Egger
RWTH Aachen



Frank Hölzle	Rainer Röhrig	Behrus Puladi
🚀 How to Participate
Step 1. Sign up to Synapse
Step 2. Join or Create a Team
Step 3. Register for the Challenge
Step 3. Access the Data
Step 4. Prepare your Docker
Step 5: Submit
Step 6. Share Ideas and Ask Questions

The challenge uses this site, Synapse, for all related efforts. To learn more about using Synapse, visit our documentation. A good place to start is the Getting Started guide. To learn about what Synapse can be used for, please read the Synapse FAQ.

Step 1. Sign up to Synapse
Login or register -- Only an email address is required. Learn more about Synapse accounts.

Step 2. Join or Create a Team
It is encouraged, but not required, to work on this challenge as a team.

Create and Register a Team
Learn more about how to create a team. By default, the participant who creates a team is the Team Captain and has the ability to invite and remove members. All team members need a Synapse account and must log into Synapse and accept the team invitation.


Join an Existing Team
You may also request to join an existing team by requesting to join a team registered for the challenge. Visit the Participants and Teams page to see which teams have registered. Learn more about joining a team.

Step 3. Register for the Challenge
By registering, you agree to all the Rules.


You have successfully registered for the Challenge.

You must agree to the Challenge Terms to register.

Click here to see where the world solvers are coming from.

Step 3. Access the Data
Access to the data of the challenge will only be provided once the following form (step 2) has been completed.

Learn more about the Data available.



Step 4. Prepare your Docker
There are two parts to the challenge submission for you to be eligible.

Your Docker.
Your Write-Up
To upload your docker or other files to Synapse, you must be a Certified User.


Step 5: Submit
Make sure to indicate whether you are submitting as an individual or a team. If submitting as a team, identify the team. Once you have submitted as a Team, you may not submit as an individual and vice versa.

Docker submission:

docker name: oss25_<team_name>
docker tag: v<version_number> (please use versioning, i.e. v1, v2, v3)
Please submit each update to the evaluation queue to be evaluated
For the writeup submission, AFTER your final challenge submission, please share your private project with the public with Can Download permissions. Read more here about how to share things in Synapse.

Make sure to share (under project sharing settings) the projects for the docker and the writeup with us (OSS Challenge Admin), so we receive a notification.
Helping the evaluation process run smoothly



After you submit, expect an email either indicating that your project was submitted successfully or that your entry is considered invalid because of specific problems with your entry. If there is any problem with your submitted entry, a description of the problems (e.g. missing items) will be sent to you. Use this information to modify and resubmit your entry. These emails will be sent to the email address saved in your synapse profile (if you have not changed it, this is the email you used for registration). If you do not receive this email, check your spam folder.

Step 6. Share Ideas and Ask Questions
The discussion forum should be used for:

Any and all questions about the challenge
Collaboration with other participants.
Learn more about discussion forums.
⚖️ Competition Guidelines
Please Review the Detailed Challenge Description
(starting on page 37)
🚨 Challenge Terms of Use
Participants must register before the submission deadline.
To ensure fair competition and comparability only public datasets are allowed in addition to the challenge dataset (Adding labels to the challenge dataset is considered the use of private data, and such submissions will be disqualified)
All code used to generate results must be included in the Docker container.
Only complete docker submissions will be considered
Late submissions will not be accepted. No exceptions unless due to technical issues - please contact the organizer immediately.
Teams must disclose any external datasets used, and they must be cited
Please also note the rules on the Endoscopic Vision Challenge main page
Access to the data of the challenge will only be provided once the following form has been filled out.
⚙️ Technical Requirements
Submissions must produce output in the required format
Docker containers must run without internet access
Methods must be fully automatic (no user interaction)
🧑‍⚖️ Evaluation & Integrity
Only the final submission of each team will be evaluated
Participants from the organizers' groups are welcome to join and can have their results featured in publications and on the leaderboard. However, they are not eligible for awards.
Any attempt to exploit the evaluation server or leak test labels or data will result in disqualification
The organizers reserve the right to request clarification or additional information about any submission
📢 Publication Policy
We will collaborate with participants to create a comprehensive journal article summarizing the key results and analyses from this challenge. Participants who submit valuable work are welcome to contribute to the publication.

In order for us to include you in our paper:

Please submit a detailed description of your solution with your final test phase submission.
Use the template and follow the instructions we provided
Before publication of the joint paper, no results may be published.

🧩 Tasks
The goal of this challenge is to determine the best model for video-based skill assessment for open suturing.

Please note: All tasks will take a directory with video files (.mp4) as input. Please develop your evaluation scripts accordingly.

Task 1
Classify the global rating score (GRS) into four classes (0-3, integers) (novice (class 0): 8-15, intermediate (class 1): 16-23, proficient (class 2): 24-31, expert (class 3): 32-40)

Output Format
The submitted docker should output a csv file in the following format (include the headers):

VIDEO	GRS
Task 2
Classify the five different scores (0-4, integers) in the eight different objective structured assessment of technical skill (OSATS) categories

Output Format
The submitted docker should output a csv file in the following format (include the headers):

VIDEO	OSATS_RESPECT	OSATS_MOTION	OSATS_INSTRUMENT	OSATS_SUTURE	OSATS_FLOW	OSATS_KNOWLEDGE	OSATS_PERFORMANCE	OSATSFINALQUALITY
Task 3
Track hands and tools using keypoints

Output Format (MOTChallenge-modified)
The submitted docker should output a csv file per video in the following format (no headers), one row per tracked object:

Frame	Track ID	Class ID	Bbox xywh	KP xyc
Frame	Frame number
Track ID	Track ID of object
Class ID	Class ID of object (see Annotation Details below)
Bbox xywh	Bounding box information separated by commas x, y, width, height (optional or set as -1 for compatibility)
KP xyc	Keypoints separated by commas x, y, confidence (please note that hands and tool KP amounts differ)


🗂️ Data
Task 1 & 2: Classification
The data consists of videos of medical and dental students and residents undergoing open surgical suturing training. Videos are recorded from a bird's-eye-view perspective, are approx. 5 mins in length, and are rated by three individual raters. Labels (GRS and OSATS) for each video are collected in an Excel spreadsheet along with the corresponding participant ID. Each student was recorded twice: once before and once after a training program. Surgical residents were recorded once.


(Please note that the videos saved in the Files section of this project are not the complete dataset. Please use the above button to download the full dataset for Task 1 and 2)

Reference
Annotations are saved in OSATS_MICCAI_trainset.xslx with the following format.

STUDENT	GROUP	TIME	SUTURES	INVESTIGATOR	VIDEO	OSATS_RESPECT	OSATS_MOTION	OSATS_INSTRUMENT	OSATS_SUTURE	OSATS_FLOW	OSATS_KNOWLEDGE	OSATS_PERFORMANCE	OSATSFINALQUALITY	GLOBARATINGSCORE
Descripiton
STUDENT	Unique identifier for the person performing in the Video
GROUP	Cohort. Note: this category is included just for completion or for you to use when loading the dataset; it will NOT be used to determine the GRS class for Task 1 (please see Task 1 for more information).
TIME	Each student is recorded twice (between an instruction session) and has a pre and post video. Experts only have a post video
SUTURES	Number of sutures made. Note: this category is included just for completion and will NOT be used in classification
INVESTIGATOR	Rater: A - F, each case is only rated by three raters
VIDEO	Unique video ID, matches mp4 file name
OSATS_X	OSATS category score (1-5) (see Task 2 for more information)
GLOBARATINGSCORE	GRS score (8-40) (see Task 1 for more information)
Task 3: Tracking
It is not permitted to train on the validation set
Methods that do so will be automatically disqualified


Train Set
The data consists of frames from ~one minute long clips of videos from the Task 1 & 2 Dataset. Frames are sampled at 1 frame per minute. Annotations include segmentation masks of hands and tools, as well as keypoints: hands (5 keypoints), tools (3 keypoints).

⚠️ Covered objects may have keypoints annotated, but no segmentation masks ⚠️

Structure
train/
├── frames/
│ ├──<video_ID>_frame_<frame_ID>.png
│ └── ...
├── masks/
│ ├──<video_ID>_frame_<frame_ID>_<tool_name>_mask.png
│ ├── <video_ID>_frame_<frame_ID>_<tool_name><suffix>_mask.png
│ └── ...
└── mot/
│ ├── <video_ID>_frame_<frame_ID>.txt
│ └── ...

⚠️⚠️ Some frames have more than one needle (max. 2). Those with multiple instances have a suffix starting with index 1 (i.e. the mask for the second needle would contain "needle1" in the name)

Reference
Column headers are not included in file.

Frame	Track ID	Class ID	Bbox xywh	KP xyv
Description
Frame	Frame number
Track ID	Track ID of object
Class ID	Class ID of object (see Annotation Details below)
Bbox xywh	Bounding box information separated by commas x, y, width, height (set as -1 because not used, compatibility with MOT format)
KP xyv	Keypoints separated by commas x, y, visibility (please note that hands and tool KPs differ)
Validation Set
To enable better validation we have provided a separate validation set which may only be used for validation purposes. Similarly to the Train set, the data are frames from 1 minute clips sampled from the Task 1 & 2 Dataset. In contrast for better validation, frames are sampled at 1 fps. Annotations only include keypoints of hands (5 keypoints) and tools (3 keypoints).

Structure
⚠️⚠️ Frame structure differs from train set!!
val/
├── frames/
│ ├──<video_ID>/
│ │ ├──<video_ID>_frame_<frame_ID>.png
│ │ └── ...
│ └── ...
└── mot/
│ ├── <video_ID>_frame_<frame_ID>.txt
│ └── ...

Reference
Description
See Train Set Description and Reference.

Annotation Details
Segmentation Classes
0	Left Hand
1	Right Hand
2	Scissors
3	Tweezers
4	Needle Holder
5	Needle
Keypoint Details
Hands

0	Thumb
1	Middle
2	Index
3	Ring
4	Pinky
5	Back of hand
Tools

Scissors	0	Left (Sharp point)
1	Right (Broad point)
2	Joint
Tweezers	0	Left (with Nub)
1	Right (with Hole)
2	Nub
Needle Holder	0	Left
1	Right (Right when text visible)
2	Joint
Needle	0	Left (End)
1	Right (Tip)
2	Middle
Visbility Flags

0	Out of Frame
1	Hidden
2	Visbile

📊 Evaluation
Only full submisisons of each task will be considered. Only the final submission of each team will be evaluated.

For more details, please visit our ⚖️ Competition Guidelines

📐 Metrics
Task 1 & 2:
F1-Score (Dice Similarity Coefficient)
Expected Cost
Rater annotations will be averaged.
Rankings for individual metrics will be determined given the order and averaged to one rank per task.

Here is the code we will use to calculate the metrics.

Task 3:
HOTA adapted for keypoints
Here is the code we will use to calculate the metrics.
As we are still finalizing the scripts, please use the branch 'devel-kp'. The script to run is scripts/run_mot_challenge_kp.py.
Data file structure:
data/
├── gt/
│ ├──<video_ID>txt
│ └── ...
├── trackers/
│ ├──<video_ID>_pred.txt
│ └── ...

📟 Scripts
Task 1 & 2 metrics
Task 3 metrics -- please use the "devel-kp" branch!!!

🏆 Challenge Results
Results will be announced on the challenge day.

Check the 📅 Timeline for more details.

💡 Help & Templates
🐋 Docker Instructions
Docker Instructions
The challenge consists of 3 tasks. To participate in a given task, a team must provide a docker image that takes the described input and produces the described output. A docker container that solves multiple tasks can also be submitted. Please note below how it should be called correctly.

👣 Walk-through: How to create and submit your docker
We have created a detailed walk-through on how to create the required NVIDIA Docker image for the submission. If any problems or questions arise, please contact us via the 💬 Discussion Forum, so others may also benefit from your inquiries.


🔢 Docker Container for Multiple Tasks
The docker should expect an input of the following format:

docker run --network none --rm --gpus '"device=0"' --ipc=host -v "<Input folder>/:/input:ro" -v "<Output folder>:/output" <docker name> /usr/local/bin/Process_SkillEval.sh <task type>
Here <input folder> and <output folder> will be folders on the host PC. <docker name> is the name of the docker image. The shell script will be executed, and it will be passed the input directory containing the test set videos (mp4 format) and the <task type> ('GRS', 'OSATS', or 'TRACK').

Task 1 GRS score: The container should then produce a valid GRS score within a CSV file formatted as follows (please include the column headers):
VIDEO	GRS
Task 2 OSATS score: The container should produce a list of valid OSATS scores within a CSV file formatted as follows (please include the column headers):
VIDEO	OSATS_RESPECT	OSATS_MOTION	OSATS_INSTRUMENT	OSATS_SUTURE	OSATS_FLOW	OSATS_KNOWLEDGE	OSATS_PERFORMANCE	OSATSFINALQUALITY
Task 3 Tracking: The container should produce a list of tracks in MOTChallenge-modified formatted as follows (no headers):
Frame	Track ID	Class ID	Bbox xywh	KP xyc
Frame	Frame number
Track ID	Track ID of object
Class ID	Class ID of object (see Annotation Details below)
Bbox xywh	Bounding box information separated by commas x, y, width, height (optional or set as -1 for compatibility)
KP xyc	Keypoints separated by commas x, y, confidence (please note that hands and tool KP amounts differ)
 

1️⃣ Docker Container for Single Task
The input is similar to before, just no information regarding the task will be given. The shell script should directly execute the chosen task. The container will be called with:

docker run --network none --rm --gpus '"device=0"' --ipc=host -v "<Input folder>/:/input:ro" -v "<Output folder>:/output" <docker name> /usr/local/bin/Process_SkillEval.sh
Here <input folder> and <output folder> will be folders on the host PC. <docker name> the name of the docker image.

Task 1 GRS score: The container should then produce a valid GRS score within a CSV file formatted as follows (please include the column headers):
video_name	GRS
Task 2 OSATS score: The container should produce a list of valid OSATS scores within a CSV file formatted as follows (please include the column headers):
video_name	OSATS_RESPECT	OSATS_MOTION	OSATS_INSTRUMENT	OSATS_SUTURE	OSATS_FLOW	OSATS_KNOWLEDGE	OSATS_PERFORMANCE	OSATSFINALQUALITY
Task 3 Tracking: The container should produce a list of tracks in MOTChallenge-modified formatted as follows (no headers):
Frame	Track ID	Class ID	Bbox xywh	KP xyc
Frame	Frame number
Track ID	Track ID of object
Class ID	Class ID of object (see Annotation Details below)
Bbox xywh	Bounding box information separated by commas x, y, width, height (optional or set as -1 for compatibility)
KP xyc	Keypoints separated by commas x, y, confidence (please note that hands and tool KP amounts differ)
✍️ Write-Up Instructions
1. Use the provided template
Follow the structure and formatting of the official method description template.
Make sure to include:

Team name
Method overview
Architecture/technique details
Training data and preprocessing
Inference details
Postprocessing (if any)
Performance highlights (optional)
2. Submit your writeup
Option A (preferred): Add your method write-up to your project wiki page.
Submit any part of your project page (any file or docker) to the write-up evaluation queue
Option B: Submit your write-up as a PDF.
Upload the PDF to you project's "Files" section
Submit it to the write-up evaluation queue
Please remember to share your project with the Open Suturing Skills Challenge Admin
3. Include visual content
Upload all images used in your write-up in original resolution to the “Files” section of your project page.

4. Deadline
Submit or update your write-up by September 15th, 2025, 11:59 PM GMT
👥 Participants & Teams
📓 Templates
❔ Help
We encourage you to check out our 💬 Discussion Forum for help. Maybe your question is already answered. If not, feel free to open a new thread, so others may benefit form your question as well.