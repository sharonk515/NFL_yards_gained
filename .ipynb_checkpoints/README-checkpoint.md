# Understanding NFL Yard Gain from Handoff Plays

Project led by Sharon Kwak

“The running back takes the handoff… he breaks a tackle…spins… and breaks free! One man to beat! Past the 50-yard-line! To the 40! The 30! He! Could! Go! All! The! Way!”

But will he?

American football is a complex sport. From the 22 players on the field to specific characteristics that ebb and flow throughout the game, it can be challenging to quantify the value of specific plays and actions within a play. Fundamentally, the goal of football is for the offense to run (rush) or throw (pass) the ball to gain yards, moving towards, then across, the opposing team’s side of the field in order to score. And the goal of the defense is to prevent the offensive team from scoring.

In the National Football League (NFL), roughly a third of teams’ offensive yardage comes from run plays. Ball carriers are generally assigned the most credit for these plays, but their teammates (by way of blocking), coach (by way of play call), and the opposing defense also play a critical role. Traditional metrics such as ‘yards per carry’ or ‘total rushing yards’ can be flawed; in this competition, the NFL aims to provide better context into what contributes to a successful run play.

As an “armchair quarterback” watching the game, you may think you can predict the result of a play when a ball carrier takes the handoff - but what does the data say? Deeper insight into rushing plays will help teams, media, and fans better understand the skill of players and the strategies of coaches. It will also assist the NFL and its teams evaluate the ball carrier, his teammates, his coach, and the opposing defense, in order to make adjustments as necessary.

## Goals

Analyze which features most impact the number of yards gained by a player during a handoff (when a Quarterback hands the ball to a Running Back).

## Technology Used
- Python
- Pandas
- Statsmodels
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- CatBoost

## Data

Data was pulled from [Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2020/data), which consists of data from 2017 and 2018 seasons. Each row is information for each handoff play, totaling 509762 plays, by a specific player with 37 different features.

## Strategy

**Regression**

Run a trained regression model to find the most important features in predicting the number of yards gained.

- Linear Regression (scikit-learn and statsmodels)
- Lasso
- Ridge
- Random Forest Regression
- XGBoost Regression

**Classification**

Using yards grouped into various ranges (negative yards, 0-1, 2-3, 4-6, greater than 6), run a trained classifier to find the most important features in predicting the range of yards gained.

- Logistic Regression
- Random Forest Regression
- CatBoost Regression

**National Football League**

[Offense](https://operations.nfl.com/football-101/formations-101/)
<br> 11 players who try to move the football down the field to either get into its opponent's end zone or get close enough to attempt to kick a field goal.
- **QB** Quarterback: leads the offense and calls the plays, leads team down the field by running with the ball, handing the ball off or completing a forward pass to an eligible receiver (Wide Receiver, Running Back, Fullback, Tight End)
- **RB** Running Back (halfback or tailback): primary ball carrier who runs with the ball, tries to catch a pass, or remains in the backfield to block for the quarterback
- **FB** Fullback: blocks for the running back or quarterback, carry the ball when a strong running style is needed (e.g. when offense only needs to gain a few yards for a first down or to score a touchdown)
- **RB** Wide Receiver: known for speed and ability to catch the ball; lines up close to the sidelines and runs downfield to catch passes from the quarterback
- **TE** Tight End: lines up on the end of the offensive line as an extra blocker on running plays or a receiver in passing plays
- Linemen: heroes of offense
    - **C** Center: lines up in middle and snaps football between his legs to the quarterback to start each play, communicating the blocking scheme to other linemen
    - **G** Guard: lines up on either side of center blocks oncoming defenders on passing plays or tries to open running lanes for the running back on rushing plays
    - **T** Tackles: lines up outside the guards, joining in pass protection and run blocking

![offense](Images/Offense.png "Offense")

[Offense Formations](https://protips.dickssportinggoods.com/sports-and-activities/football/football-101-football-formations)
- SINGLEBACK: uses one Running Back behind the Quarterback, lined up directly behind or offset to either side
- SHOTGUN: Quarterback positions himself about 5-7 yards behind the line of scrimmage, requiring the Center to throw the ball
- I_FORM: 2 Wide Receivers, 2 Running Backs, 1 Tight End; Running Backs behind the quarterback, one behind the other, resembling an "i" dotted by the Quarterback; Wide Receivers split out wide with one on the line of scrimmage and one off the line of scrimmage; Tight End is next to a tackle (side with the Tight End is called the strong side with the opposite side being the weak side)
- PISTOL: Quarterback lines up 4 yards behind the center
- JUMBO (GOAL LINE): used in short-yardage situations, especially near the goal line, to score by brute force; 3 Tight Ends, 2 Running Backs and no Wide Receivers (or 2 Tight Ends and 3 Running Backs with no Wide Receivers)
- WILDCAT: running back or receiver who runs well takes the place of the Quarterback in a shotgun formation while the Quarterback lines up wide or is replaced by another player
- EMPTY: all of the backs play near the line of scrimmage as extra wide receivers or tight ends

[Defense](https://operations.nfl.com/football-101/formations-101/)
<br> 11 players who try to keep the offense from advancing down the field and scoring.
- Defensive line:
    - **DT** Defensive Tackles and **DE** Defensive Ends: line up opposite the offensive linemen and try to push their way across the line of scrimmage (line of play) into the offensive backfield to disrupt or stop a play at or near the line of scrimmage, tackle the ball carrier on rushing plays or rush the quarterback.
- **LB** Linebacker: positioned 3-5 yards behind the defensive line and supports the defensive line by tackling the ball carrier, dropping into pass coverage or rushing the quarterback.
- Secondary (Defensive Backs):
    - **CB** Cornerback: lines up near the line opposite wide receivers
    - **S** Safety: positioned 10-15 yards downfield, last line of defense between the offense and the end zone.
    - pursue the ball carrier and sometimes rush the quarterback

![Defense](Images/Defense.png "Defense")