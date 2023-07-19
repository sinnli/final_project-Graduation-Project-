# final_project-Graduation-Project-
Simlating Ad hoc wireless networks with nodes constantly in motion is essentialy to 5G technologies.
The Ad hoc network itself can be costumized in the adhoc.py file.
There number of nodes, flows and more can changed.
There is an agent responsible for each of the flows which is always at the "head node" of the flow.
The agent isself uses DDQN in order to find the best next hop according to latency and reach criterias.


## How to run the project
There are two options; either by the main of the project which contains the train part of the 
model used by the agents, or by the evaluation file. 

The evaluation file contains another routing improvment which is done by bypassing the bottlneck
if one is detected. This done with help of "alternative agents" which temporary agents.
