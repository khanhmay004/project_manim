from sympy import symbols, Function, Sum, lambdify, diff
import pandas as pd
from math import sqrt

from manim import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model

def create_train_test_model() -> tuple:
  # Import data from a CSV file
  df = pd.read_csv('https://bit.ly/3TUCgh2', delimiter=",")

  # Initialize variables for independent (X) and dependent (Y) variables
  X = df.values[:, :-1]
  Y = df.values[:, -1]

  # Split the dataset into training and testing sets (2/3 train, 1/3 test)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=7)

  # Fit a Linear Regression model on the training set
  model = LinearRegression()
  fit = model.fit(X_train, Y_train)

  # Calculate the R^2 Score on the test set
  result = model.score(X_test, Y_test)

  # Store the coefficients of the model into m and b using ValueTrackers
  m = ValueTracker(fit.coef_.flatten()[0])
  b = ValueTracker(fit.intercept_.flatten()[0])

  # Set up the coordinate axes for visualization
  ax = Axes(
      x_range=[0, 100, 20],
      y_range=[-40, 200, 40],
      axis_config={"include_tip": False},
  )

  # Plot the training and testing data points
  train_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                  pd.DataFrame(data={'x': X_train.flatten(), 'y': Y_train.flatten()}).itertuples()]
  test_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                  pd.DataFrame(data={'x': X_test.flatten(), 'y': Y_test.flatten()}).itertuples()]

  # Plot the linear regression line
  line = Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value())).set_color(YELLOW)

  # Make the line follow changes in the m and b values
  line.add_updater(
      lambda l: l.become(
          Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value()))).set_color(YELLOW)
  )

  # Return the ValueTrackers for m and b, coordinate axes, and visual elements for training, testing points, and the line
  return m, b, ax, train_points, test_points, line

class TrainTestScene(Scene):

    def construct(self):
        # Create linear regression model and visual elements
        m, b, ax, train_points, test_points, line = create_train_test_model()

        # Group training and testing points for easy manipulation
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)

        # Add axes and data points to the scene
        self.add(ax, train_group, test_group)

        # Create visual elements for training and testing split boxes
        training_rect = Rectangle(color=BLUE, height=1.5, width=1.5, fill_opacity=.8).shift(LEFT * 4 + UP * 2.5)
        training_text = Text("TRAIN").scale(.6).move_to(training_rect)
        training_rect_grp = VGroup(training_rect, training_text)

        test_rect = Rectangle(color=RED, height=1.5 / 2, width=1.5, fill_opacity=.8).next_to(training_rect, DOWN)
        test_text = Text("TEST").scale(.6).move_to(test_rect)
        test_rect_grp = VGroup(test_rect, test_text)

        # Wait for a moment
        self.wait()

        # Animate the filling of testing points with red color
        self.play(*[p.animate.set_fill(RED) for p in test_points])
        self.wait()

        # Copy training and testing points for reverse transformation
        train_points_copy = [p.copy() for p in train_points]
        test_points_copy = [p.copy() for p in test_points]

        # Animate the transition from data points to training and testing split boxes
        self.play(ReplacementTransform(test_group, test_rect_grp))
        self.wait()
        self.play(ReplacementTransform(train_group, training_rect_grp))

        # Create braces and labels for training and testing split ratios
        b1 = Brace(test_rect, direction=RIGHT)
        b1_label = MathTex(r"\frac{1}{3}").scale(.7).next_to(b1, RIGHT)
        self.play(Create(b1), Create(b1_label))

        b2 = Brace(training_rect, direction=RIGHT)
        b2_label = MathTex(r"\frac{2}{3}").scale(.7).next_to(b2, RIGHT)
        self.play(Create(b2), Create(b2_label))

        # Wait for a moment
        self.wait()

        # Fade out braces and labels
        self.play(FadeOut(b1), FadeOut(b2), FadeOut(b1_label), FadeOut(b2_label))
        self.wait()

        # Group copied training and testing points
        train_points_copy_grp = VGroup(*train_points_copy)
        test_points_copy_grp = VGroup(*test_points_copy)

        # Animate the transition back to data points from training and testing split boxes
        self.play(ReplacementTransform(training_rect_grp, train_points_copy_grp))
        self.wait()

        # Create and animate the linear regression line
        self.play(Create(line))
        self.wait()

        # Fade out copied training points
        self.play(FadeOut(train_points_copy_grp))

        # Animate the transition back to data points from testing split box
        self.play(ReplacementTransform(test_rect_grp, test_points_copy_grp))
        self.wait()

def create_problem_model() -> tuple:
    # Read data from the provided CSV URL into a DataFrame
    df = pd.read_csv("https://bit.ly/45VEJfL", delimiter=",")

    # Extract features (X) and target variable (Y) from the DataFrame
    X = df.values[:, :-1]  # area
    Y = df.values[:, -1]  # price

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=49)

    # Create a DataFrame for the training data
    train_data = pd.DataFrame(data=X_train, columns=df.columns[:-1])
    train_data['y'] = Y_train
    train_data = list(train_data.itertuples())  # Convert DataFrame rows to namedtuples

    # Fit a linear regression model to the training data
    model = LinearRegression().fit(X_train, Y_train)

    # Use ValueTracker to track the R^2 score of the model on the test set
    result = ValueTracker(model.score(X_test, Y_test))

    # Use ValueTrackers to track the slope (m) and intercept (b) of the model
    m = ValueTracker(model.coef_.flatten()[0])
    b = ValueTracker(model.intercept_.flatten()[0])

    # Set up a coordinate system (axes) for the visualization
    ax = Axes(
        x_range=[0, 120, 10],
        y_range=[0, 210, 50],
        axis_config={"include_tip": True},
    ).add_coordinates()

    # Add axis labels to the coordinate system
    labels = ax.get_axis_labels(
        MathTex(r"\text{Size of House (m}^2\text{)}").scale(0.7),
        MathTex(r"\text{Price of House (million VND)}").scale(0.7)
    )

    # Create dots representing training and testing data points on the coordinate system
    train_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                    pd.DataFrame(data={'x': X_train.flatten(), 'y': Y_train.flatten()}).itertuples()]

    test_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                   pd.DataFrame(data={'x': X_test.flatten(), 'y': Y_test.flatten()}).itertuples()]

    # Create a line representing the linear regression model
    line = Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(100, m.get_value() * 100 + b.get_value())).set_color(
        YELLOW)
    line.add_updater(
        lambda l: l.become(
            Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(100, m.get_value() * 100 + b.get_value()))).set_color(
            YELLOW)
    )

    # Return the generated elements as a tuple
    return train_data, m, b, ax, train_points, test_points, line, labels, result

def sse_fit(scene, data, m, b, ax, points, line) -> tuple:
    # List to store residuals (lines representing the vertical distance between data points and the regression line)
    residuals: list[Line] = []

    # Iterate through each data point
    for d in data:
        # Create a residual line for the current data point and animate its appearance
        residual = Line(start=ax.c2p(d.x, d.y), end=ax.c2p(d.x, m.get_value() * d.x + b.get_value())).set_color(RED)
        scene.play(Create(residual), run_time=.2)

        # Add an updater to dynamically adjust the residual line during animation
        residual.add_updater(lambda r, d=d: r.become(
            Line(start=ax.c2p(d.x, d.y), end=ax.c2p(d.x, m.get_value() * d.x + b.get_value())).set_color(RED)))
        residuals += residual  # Add the residual line to the list

    # Function to calculate the Sum of Squared Errors (SSE) given current values of slope (m) and intercept (b)
    def get_sse(m, b):
        new_sse = ValueTracker(0.0)
        for i in range(len(residuals)):
            new_sse += (residuals[i].get_length() ** 2)
        return new_sse

    # Return the list of residuals and the SSE calculation function
    return residuals, get_sse



class TableExamples(Scene):
    # Main function for constructing the scene
    def construct(self):
        # Create title and subtitle text objects
        title = Text("Linear Regression")
        subTitle = Text("Hanoi Housing Price Problems").set_color_by_gradient(BLUE, GREEN)

        # Create problem statement text objects
        problem = Text("Given the historical house selling prices in the table,", font_size=32).to_edge(UP).set_color(YELLOW)
        problem2 = Text("can you find the optimal price of a 70 m² house?", font_size=32).next_to(problem, DOWN, buff=0.2).set_color(YELLOW)

        # Adjust title and subtitle positions
        title.shift(0.5 * UP)
        subTitle.next_to(title, DOWN)

        # Animate the appearance and movement of title and subtitle
        self.play(Write(title), Write(subTitle))
        self.play(ApplyMethod(title.shift, 3*UP), FadeOut(subTitle), FadeOut(title))

        # Animate the appearance of problem statements with a lagged start
        self.play(LaggedStart(
            Write(problem),
            Write(problem2),
            lag_ratio=0.4,
            run_time=3
        ))

        # Create a historical data table
        t2 = Table(
            [["30", "48"],
            ["40", "62"],
            ["50", "75"]],
            col_labels=[Text("Area"), Text("Price")],
            include_outer_lines=True
        )

        # Set colors for column labels in the historical data table
        lab_2 = t2.get_col_labels()
        for item in lab_2:
            item.set_color(YELLOW)

        # Create outline for the historical data table and scale it
        outline_2 = VGroup(t2.vertical_lines, t2.horizontal_lines)
        t2.scale(0.5).shift(LEFT*4)

        # Create a prediction data table
        t1 = Table(
            [["30", "48"],
            ["40", "62"],
            ["50", "75"],
            ["70", "?"],
            ["120", "?"],
            ["150", "?"]],
            col_labels=[Text("Area"), Text("Price")],
            include_outer_lines=True
        )

        # Set colors for column labels in the prediction data table
        lab_1 = t1.get_col_labels()
        for item in lab_1:
            item.set_color(YELLOW)

        # Create outline for the prediction data table and scale it
        outline_1 = VGroup(t1.vertical_lines, t1.horizontal_lines)
        t1.scale(0.5).shift(RIGHT*4)

        # Get rows from the prediction data table
        row_1 = t1.get_rows()
        row1to3 = VGroup(row_1[1], row_1[2], row_1[3])
        rowlast = VGroup(row_1[4], row_1[5], row_1[6])

        # Set color for the last row in the prediction data table
        rowlast.set_color(BLUE)

        # Define mathematical equations for linear regression
        eq1 = MathTex("Area", "=?.Price+?").next_to(t1, DOWN, buff=1)
        eq2 = MathTex(r"y", r"= \beta_1", "x", r"+ \beta_0").next_to(t1, DOWN, buff=1)
        eq3 = MathTex("y", "=", "mx", "+", "b").next_to(t1, DOWN, buff=1)

        # Create a framebox around the first equation
        framebox1 = SurroundingRectangle(eq1)

        # Create arrows to represent the flow of information
        arrow_1 = Arrow(start=t2.get_right(), end=t1.get_left()).set_angle(0.2)
        arrow_2 = Arrow(start=t2.get_right(), end=t1.get_left(), color=BLUE).set_angle(-0.2)
        arrow_3 = Arrow(start=t1.get_bottom(), end=framebox1.get_top(), color=YELLOW).scale(1.5)

        # Create braces with accompanying text for historical data and prediction
        b1 = Brace(arrow_1, direction=arrow_1.copy().rotate(PI / 2).get_unit_vector())
        b1text = b1.get_tex(r"\text{Historical data}").scale(0.7)
        b2 = Brace(arrow_2, direction=arrow_2.copy().rotate(-PI / 2).get_unit_vector(), color=BLUE)
        b2text = b2.get_tex(r"\text{Prediction}").scale(0.7)
        b1_comb = VGroup(b1, b1text)
        b2_comb = VGroup(b2, b2text)

        # Animate the creation of outline, column labels, and arrows for the historical data table
        self.play(Create(outline_2))
        self.play(Write(t2.get_columns()[0]))
        self.wait()
        self.play(Write(t2.get_columns()[1]))
        self.wait()

        # Animate the creation of outline, column labels, arrows, and braces for the prediction data table
        self.play(Create(outline_1))
        self.play(Write(t1.get_labels()))
        self.play(LaggedStart(Create(arrow_1), Create(b1_comb)))
        self.play(Write(row1to3))
        self.wait(1)
        self.play(LaggedStart(Create(arrow_2), Create(b2_comb)))
        self.play(Write(rowlast))
        self.play(Create(arrow_3))
        self.wait()

        # Animate the appearance of the first equation with a framebox
        self.play(FadeIn(eq1), Create(framebox1))
        self.wait(1)

        # Animate the transition between equations
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)
        self.play(ReplacementTransform(eq2, eq3))
        self.wait()

class FitScene2(Scene):
    def construct(self):
        # Create a linear regression problem and obtain necessary elements
        train_data, m, b, ax, train_points, test_points, line_f, labels, result = create_problem_model()

        ## Show dataset (both data + train)
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)
        self.play(Create(ax), Write(labels))
        self.play(Create(train_group), Create(test_group))
        self.wait()

        # Comment on the appearance of the graph and mention the use of a linear regression model
        # Split the dataset into training and testing sets
        self.play(*[p.animate.set_fill(RED) for p in test_points])
        self.wait()
        self.play(FadeOut(test_group))
        self.wait()

        # Visualize the initial linear regression model (bad fit)
        m_1 = ValueTracker(0)
        b_1 = ValueTracker(100)
        SSE_of_best_func = ValueTracker(12.781)
        eq1 = always_redraw(lambda : MathTex(f"y = {m_1.get_value():.3f}", r"x + ", f"{b_1.get_value():.3f}").move_to((LEFT*3 + UP*2.5)))
        line_1 = Line(start=ax.c2p(0, b_1.get_value()), end=ax.c2p(100, m_1.get_value() * 100 + b_1.get_value())).set_color(YELLOW)
        line_1.add_updater(
            lambda l: l.become(
                Line(start=ax.c2p(0, b_1.get_value()), end=ax.c2p(100, m_1.get_value() * 100 + b_1.get_value()))).set_color(YELLOW)
        )
        sse_text_1 = MathTex("SSE = ").next_to(eq1,DOWN,buff=0.3)

        # Animate the creation of the initial model and SSE visualization
        self.play(Create(line_1))
        self.play(Create(eq1),Write(sse_text_1))
        residuals_b, get_sse = sse_fit(self, train_data, m_1, b_1, ax, train_points, line_1)
        sse_1 =  get_sse(m_1, b_1)
        sse_text_2 = always_redraw(lambda : MathTex("SSE = ",f'{sse_1.get_value():.3f}').next_to(eq1,DOWN,buff=0.3))
        self.play(ReplacementTransform(sse_text_1, sse_text_2))
        self.wait(2)

        # Optimize the model to the best fit using interpolation
        self.play(m_1.animate.set_value(m.get_value()), b_1.animate.set_value(b.get_value()), sse_1.animate.set_value(SSE_of_best_func.get_value()), run_time=3, rate_func=linear)
        self.wait(2)

class FinalExample2(Scene):
    def construct(self):
        # Create a linear regression problem and obtain necessary elements
        data, m, b, ax, train_points, test_points, line_f, labels, result = create_problem_model()
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)

        # Display the training and testing data points, along with the initial linear regression model
        eq1 = MathTex(f"y = {m.get_value():.3f}", r"x + ", f"{b.get_value():.3f}").move_to(LEFT*2.5 + UP*2.3)
        r2_score = MathTex("R^2 \\text{ Score: }", f"{result.get_value():.3f}").next_to(eq1, DOWN, buff=0.5).align_to(eq1, LEFT)
        r2_score[1].set_color(RED)
        test_group.set_color(RED)

        # Prediction visualization for a 70 m² house
        point = ax.coords_to_point(70, m.get_value()*70+b.get_value())
        u_point = ax.c2p(0, m.get_value()*70+b.get_value())
        price_y = MathTex(f"{m.get_value()*70+b.get_value():.0f}").next_to(u_point, LEFT, buff=0.3).set_color(RED)
        dot = Dot(point)
        ver_line = ax.get_vertical_line(point, line_config={"dashed_ratio": 0.85})
        hori_line = ax.get_horizontal_line(point, line_config={"dashed_ratio": 0.85})

        # Play the scene with animations
        self.play(Create(ax), Create(train_group), run_time=1)
        self.play(Write(labels))
        self.wait()
        self.play(Write(eq1))  # Display the best model equation
        self.play(Create(line_f), run_time=1)
        self.play(FadeOut(train_group))
        self.play(FadeIn(test_group))
        self.wait()

        # Display the R^2 score
        self.play(Write(r2_score))

        # Predict and visualize the price of a 70 m² house
        self.play(Create(hori_line), run_time=2)
        self.play(FadeIn(dot), Create(ver_line))
        self.wait(1)
        self.play(FadeIn(price_y))
        self.wait()

        # Create a left panel containing relevant elements and scale it for better visibility
        left_panel = VGroup(ax, train_group, labels, eq1, line_f, test_group, r2_score, hori_line, dot, ver_line, price_y).scale_to_fit_height(5)

        # Display conclusion text
        conclusion_text = Text("Conclusion: The optimal price of a 70 m² house in Hanoi,").scale(0.5).to_edge(UP)
        con2 = Text("based on our Linear Regression model is 127 million VND.").scale(0.5).next_to(conclusion_text, DOWN, buff=0.3)
        self.play(
            left_panel.animate.to_edge(LEFT).to_edge(DOWN), run_time=4
        )
        self.play(Write(conclusion_text))
        self.play(Write(con2))
        self.wait()
