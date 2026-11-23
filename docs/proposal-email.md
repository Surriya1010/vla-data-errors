# Course proposal email (draft, NOT sent)

Fill in the professor's name; remove questions that are already answered.

---

**Subject:** Manipulator project proposal: Franka control and robot-learning data quality (simulation)

Dear Professor [Name],

I would like to propose a topic for my Manipulator course project and ask for
your approval before I begin.

**Proposed title:** *Franka Panda control in simulation, and how errors in
demonstration data affect a learned manipulation policy*

**1. Core (manipulator modelling and control)**
Using the Franka Panda model in NVIDIA Isaac Lab, I will:
- derive and verify the forward kinematics and Jacobian against the simulator;
- implement and compare differential inverse kinematics and operational-space
  control on reaching and pick-and-place motions;
- evaluate tracking error, behaviour near singularities and joint limits, and
  trajectory profiles.

**2. Extension (bounded experiment)**
Robot learning depends on demonstration data, and real data always contains
errors: timing offsets between camera and joint measurements, inaccurate
labels, and imprecise operator motions. In simulation the true values are
known, so these errors can be added in a controlled way. I will:
- take clean Franka demonstrations from the LIBERO benchmark;
- add one error type at a time, at a few strengths (for example a timing offset
  of 0 to 4 frames);
- fine-tune a small vision-language-action model (SmolVLA, 450M parameters) on
  each version, and measure the change in task success;
- test whether simple checks on the data can detect the errors before training.

**Scope**
- Simulation only, on my own laptop (RTX 5060, 8 GB). No lab hardware required.
- The core stands on its own. The extension is limited to 2-3 error types.
- Isaac Lab and the Franka control tutorials already run on my machine.

**Questions**
1. Is this topic suitable for the course, and is the balance between the
   modelling core and the learning extension acceptable?
2. What is the deadline, and is there a grading rubric or required format?
3. Is the project individual or in a team?
4. Would you prefer a specific robot or simulator?
5. Optional: does the lab have a Franka or another torque-controlled arm that
   students may use later?

Kind regards,
Surriya Balachandar
M.Sc. Advanced Robotics, École Centrale de Nantes
