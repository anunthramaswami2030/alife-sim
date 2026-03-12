# Artificial_life
For this project, I enhanced the traditional parallel hill climber algorithm using supervised learning. Fist, I used the vanilla parallel hill climber to collect parent and child mutations. In particular, I collected four children for every parent but still advanced each generation if best child exceeded its parent. Using this data, I trained a NN that predicts the probability that the fitness of a child will exceed that of the parent. Finally, using these probabilities, I cosntructed a modified parellel hill climber that constructs 32 mutations for each parent. Then, the NN ranks the top four and those are what is simulated. I found that this supervised guidance helped the hill climber increase the fitness of its candidates faster and lead to higher fitness atleast in the short term.

# Algorithms

<img width="1219" height="934" alt="Screenshot from 2026-03-11 23-57-59" src="https://github.com/user-attachments/assets/57c8704a-6d8a-4d93-8c11-783d67e9bddc" />

<img width="1219" height="934" alt="Screenshot from 2026-03-11 23-57-12" src="https://github.com/user-attachments/assets/ed0f8bbc-3eb3-4a04-93fb-ee9b0f3c96ae" />

<img width="1219" height="934" alt="Screenshot from 2026-03-11 23-56-04" src="https://github.com/user-attachments/assets/0e6e360d-4156-4968-aae1-6b3c4d1b50f6" />

# Plots
This plot illustrates how the fitness of the modified parellel hill climber increases faster than that of the random parallel hill climber.

<img width="1134" height="908" alt="combined" src="https://github.com/user-attachments/assets/91269a38-a57a-4a2a-a022-1188f1e30acd" />


# Video
The top robot is trained using random parellel hill climber that creates four mutations and simulates those four children per generation. The bottom robot is trained on a modified parellel hill climber that creates 32 mutations each generation and simulates the top four based on the network.

https://github.com/user-attachments/assets/779e4440-75a6-478a-a4c7-46fd93c366df


