# Introduction

The **Battle of the Algorithms** is an exhilarating two week-long coding tournament that tests and celebrates developers' skills. Eight talented participants will compete head-to-head, striving to craft the most efficient and effective algorithms. Each developer will receive a standardized User Interface (UI) and Application Programming Interface (API) to integrate their solutions.

## Rules

### Elimination Style Tournament
- Two developers will compete head-to-head in each bracket. The winner advances, and the loser is eliminated.
- Brackets will continue until the final showdown between the last two developers.
- Each bracket will have 3 rounds - best of 3 wins.

### Duration
- The competition spans two weeks, ensuring ample time for participants to develop and refine their algorithms.

### Final Competition
- At the end of the coding period, all algorithms will face off in a live tournament to determine the ultimate champion.

## Game Design

Participants will program an AI controller for the classic **Snake game**, where each snake navigates a grid to survive and gather food while avoiding obstacles and rival snakes. Developers must focus on strategic decision-making, efficient pathfinding, and adaptability to succeed.

### Gameplay Mechanics

#### Objective
- The AI snake that survives the longest while collecting food wins.
- Algorithms should prioritize survival strategies while optimizing food collection.

#### Grid & Obstacles
- The game is played on a standardized grid with randomly placed obstacles to challenge navigation.

#### Food Placement
- Food ("apples") will spawn in varying quantities and locations on the grid.
- Multiple apples can spawn at a time.
- When an apple is collected, a new one will spawn in a random location.
- There will always be exactly 3 apples on the grid, unless Sudden Death is activated.
- During Sudden Death there will only be 1 apple on the board.

#### Snake Collisions
- Developers must implement collision avoidance strategies to prevent the snake from hitting walls, obstacles, or rival snakes, while prioritizing food collection.

## Game Ending Rules

### Survival-Based
- The game ends when only one snake remains alive on the grid. This snake is declared the winner of the round.
- If multiple snakes are alive after **5 minutes**, the snake with the most food collected wins.

### Collision-Based
- If a snake collides with an obstacle, wall, or another snake, it loses - regardless of the food count.

### Time-Based
- Each round will terminate after **5 minutes**.
- If the time expires before a collision occurs, the snake with the highest food count at the time of expiration wins.

### Sudden Death Mode
- If the round ends after 5 minutes **and both snakes have exactly the same score**, the game enters **Sudden Death** mode:
    - Both snakes are temporarily paused and a "Sudden Death" message is displayed with a countdown.
    - All regular food is cleared from the grid, and a single special "sudden death apple" is spawned in a random location.
    - When the countdown ends, both snakes resume play.
    - The first snake to eat the sudden death apple wins the round.

### AI Decision Time Limit
- Each AI has a **50 milliseconds time limit** to choose a new direction for the snake's movement.
- If a direction is not chosen within this time, the snake will continue moving in the previous direction.

### Technical Failure
- If a participant’s algorithm crashes or behaves in a manner inconsistent with the rules, their snake will be disqualified unless it is proven to be a technical fault unrelated to their code.

## Algorithm Submission

To participate in the **Battle of the Algorithms**, each developer will need to submit their AI algorithm as a Python script that can be executed during a bracket. Here are the key steps for submitting and integrating your algorithm into the tournament:

### Submission Deadline
- All algorithm submissions **must be received by 3:00 PM on June 7th**.
- **Late submissions will not be accepted**, and players will be disqualified.

### Submission Method
- Code should be submitted via the [Battle of the Algorithms I - Code Submission](https://forms.gle/xHtAgJYsiFmvfpkc7) form.
- Participants must submit their AI algorithm in **Python script format**.
- The algorithm must utilize the provided **API** in the [Controller API Documentation](controller_api.md) to interact with the game environment and make decisions based on the game state.

### File Naming Convention
- To avoid confusion, all algorithm files must follow a specific **naming convention**:
  - **Format**: `name_algorithm.py` (e.g., `alice_algorithm.py`).
  - This ensures easy identification and organization of submitted algorithms.

### Algorithm Behaviour
- Please review the [Algorithm Guidelines](algorithm_guidelines.md) before diving into development.
- If an algorithm **crashes or behaves incorrectly** (e.g., infinite loops or inconsistent rules), the snake will be **disqualified** from the round unless proven to be a technical fault unrelated to the submitted algorithm.

### Testing and Debugging
- Before the tournament, it’s recommended to **test your algorithm** using the game setup provided.
- If there are errors in the final submission, the player will be notified, and each player will have a chance to **fix technical issues** after the first and only round of testing on the tournament host machine.

## Use of AI-Assisted Coding Tools
You are welcome to use AI-based tools (such as GitHub Copilot, ChatGPT, or similar assistants) to help write or optimize your algorithm. 

**However, please use AI wisely and responsibly:**
- **Understand your code:** You are responsible for any algorithm you submit. Make sure you fully understand and can explain how your code works.
- **Originality:** Do not simply copy large sections of unreviewed or unedited code from AI without verifying its correctness, efficiency, and compliance with competition rules.
- **Debugging:** Test, review, and debug any AI-generated code thoroughly before submitting.

The goal is to celebrate your problem-solving and programming skills. AI is a tool to assist—not a replacement for your own creativity and judgment.

## Competition Timeline
- 👩🏾‍💻 Coding Begins: May 25th @ 3pm
- 🗃 Coding Ends: June 7th @ 3pm
- 👾 Live Tournament: June 14th @ 6pm on Discord
- 👑 Winner Announced: June 14th (End of Live Tournament)
- 📹 Video Submissions: June  14th @ 11pm

## Video Submissions
After submissions close, each contestant will be asked to submit a short video introducing themselves and speaking about any special techniques or strategies used in their code submissions to be used in the final video on TechnicallyTi. 

Please refer to the [Video Submission Guidelines](video_submission_guidelines.md) for more details.
