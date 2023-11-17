#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sympy import symbols, Function, Sum, lambdify, diff
import pandas as pd #thư viện dùng trong phân tích dữ liệu
from math import sqrt #tính căn bậc 2 trong thư viện math 
from manim import *
from manim.mobject.geometry.tips import ArrowSquareTip
from sklearn.linear_model import LinearRegression #dùng cho hồi qui tuyến tính
from sklearn.model_selection import train_test_split ##để chia dữ liệu thành tập train và tập test
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model

#Linh
class OpeningText(Scene):
    def construct(self):
      """Phần này của mình là làm tiêu đề dẫn vào bài thì mình cần ghi tiêu đề gì thì mình cho vào Text("")
      xong đặt từng phần ra thành từng biến riêng với tên theo thứ tự. Mình ghi hết ra rồi mới cho chạy animation.
       """

      intro_words = Text("Linear Regression").set_color_by_gradient(BLUE,GREEN)# tạo tiêu đề bài làm
      introduction = Text("DSEB64A - Group 6").scale(1).set_color_by_gradient(BLUE,GREEN)

      content1 = Text("1. Linear Regression model")
      content2 = Text("2. Loss Function")
      content3 = Text("3. Training/ Testing Process")
      content4 = Text("4. Solve An Appilcation Example")

      self.play(Write(intro_words.scale(1)))
      self.play(Write(introduction.next_to(intro_words,DOWN)))
      self.wait(3)


      self.play(intro_words.animate.to_edge(UP))
      self.remove(introduction)
      self.wait(2)

      self.play(Create(content1.scale(0.6).to_edge(UL, buff= 1.75)))
      self.wait(2)

      self.play(Create(content2.scale(0.6).to_edge(UL, buff= 2.75).shift(LEFT*1)))
      self.wait(2)

      self.play(Create(content3.scale(0.6).to_edge(UL, buff = 3.75).shift(LEFT*2)))
      self.wait(2)

      self.play(Create(content4.scale(0.6).to_edge(DL, buff = 2.95).shift(LEFT*1.25)))
      self.wait(2)

class Leading(Scene):
    
    def construct(self):
        text = Text("The need of optimization tools arises from varied situations.").scale(0.6).set_color(BLUE)
        assume1 = Paragraph("Here is one example: How do you define an optimal price to\nsell your house?").scale(0.6).set_color(BLUE).to_edge(UP,buff =1 )
        assume2 = Text("Because of many limits, we will only evaluate the price based on\nthe area of the property").scale(0.6).set_color(BLUE).to_edge(UP,buff = 1)
        datatable = Table([["40","160.45"],["53","163.54"],["58","164.77"],["64","182.42"],["72","212.69"],["...","..."]], col_labels=[Text("Area"), Text("Million VND")]).scale(0.5)
        rec1 = Rectangle(width = 3.5, height = 2.0).to_edge(DOWN).set_color(YELLOW)# làm thân nhà

        roof1 = Triangle().scale(0.66)# tạo 2 tam giác để tý ghép union với 1 hình chữ nhật thành hình mái nhà
        roof2 = Triangle().scale(0.66)
        rec2 = Rectangle(width = 4.0, height = 1)
        roof1.move_to([-1,1,1])# mình phải di chuyển sao cho các hình lồng vào nhau tạo thành hình mong muốn
        roof2.move_to([3,1,4])
        rec2.move_to([1,1,3])
        roof = Union(roof1,rec2,roof2,color= RED).shift(LEFT*1).to_edge(DOWN, buff = 2.5)# union để ghép tất cả các mảnh thành hình mái nhà

        arrow1 = Arrow(start = LEFT, end = RIGHT).set_color(WHITE)

        self.play(Create(text),run_time=2)
        self.wait(1)
        self.play(text.animate.to_edge(UP,buff=1))
        self.wait(2)
        self.play(Create(VGroup(rec1,roof)))
        self.wait()
        self.play(VGroup(rec1,roof).animate.to_edge(LEFT*1.75))
        self.wait()
        self.remove(text)
        self.wait()
        self.play(Create(assume1, run_time= 2))
        self.wait(2)
        self.remove(assume1)
        self.play(Create(assume2, run_time=2))
        self.wait(2)
        self.play(Create(arrow1.to_edge(UP, buff = 2)))
        self.wait()
        self.play(Write(datatable.to_edge(UP, buff = 2).shift(RIGHT*2.75), run_time = 3))
        self.wait(2)
#Phương
def create_model() -> tuple: #hàm trả về tuple
    data = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples()) #tải dl từ file cvs vào Pandas
    m = ValueTracker(1.93939)
    b = ValueTracker(4.73333)

    ax = Axes(x_range=[0, 10],
              y_range=[0, 25, 5], #y=0,5,10,15..,
              axis_config={"include_tip": False},) # k có mũi tên

    #mô tả các điểm
    points = [Dot(point=ax.c2p(p.x, p.y), #chuyển đổi tọa độ từ hệ tọa độ của ax sang hệ tọa độ của màn hình
                  radius=.15, color=BLUE) for p in data] #từng dot là dữ liệu trong data

    # plot function (y=m*x+b)
    line = Line(start=ax.c2p(0, b.get_value()), # x=0, y= b
                end=ax.c2p(10, m.get_value() * 10 + b.get_value())).set_color(YELLOW) #x = 10, y= m*10+b

    # make line follow m and b value
    line.add_updater( # cập nhật vị trí cho line
        lambda l: l.become(
            Line(start=ax.c2p(0, b.get_value()),
                 end=ax.c2p(10, m.get_value() * 10 + b.get_value()))).set_color(YELLOW)
    )

    return data, m, b, ax, points, line #trả giá trị trong tuple

class FirstScene(Scene):

    def construct(self):
        linearr= Text("Linear Regression model", font_size=48).set_color_by_gradient(BLUE,GREEN)
        self.play(Write(linearr))
        self.wait(2)
        self.play(FadeOut(linearr))
        self.wait(2)
        #công thức
        eq1 = MathTex("y", "=m", r"x", "+b")
        eq1a = MathTex("{\hat{y}=f(x)}", "=m", r"x", "+b")
        eq2 = MathTex(r"y", r"= \beta_1", "x", r"+ \beta_0")
        eq3 = MathTex(r"y = \beta_2 x_2 + \beta_1 x_1 + \beta_0")
        eq4 = MathTex(r"y = \beta_3 x_3 + \beta_2 x_2 + \beta_1 x_1 + \beta_0")
        #box vàng
        framebox1 = SurroundingRectangle(eq1[0])
        framebox1a = SurroundingRectangle(eq1a[0])

        framebox2 = SurroundingRectangle(eq1a[2])

        #hiện công thức
        self.play(Write(eq1))
        self.wait(2)
        #chú thích
        text1 = Text("Output/Target", font_size=16, color=BLUE).next_to(framebox1, DOWN)
        text2 = Text("Input", font_size=16, color=BLUE).next_to(framebox2, DOWN)

        self.play(Write(framebox1), Write(text1))
        self.wait(2)

        self.play(ReplacementTransform(eq1, eq1a), ReplacementTransform(framebox1, framebox1a))
        self.wait(2)

        self.play(ReplacementTransform(framebox1a, framebox2), ReplacementTransform(text1, text2))
        self.wait(2)

        self.play(FadeOut(text2), FadeOut(framebox2))
        self.wait(2)

        self.play(ReplacementTransform(eq1a, eq2))
        self.wait(2)

        self.play(ReplacementTransform(eq2, eq3))
        self.wait(1)

        self.play(ReplacementTransform(eq3, eq4))
        self.wait(2)

        self.play(FadeOut(eq4))

        # vẽ biểu đồ
        data, m, b, ax, points, line = create_model() #gọi hàm creat_model
        graph = VGroup(ax, *points)

        # three versions of linear function
        eq5 = MathTex("f(x) = ", r"m ", r"x + ", "b").move_to((RIGHT + DOWN))
        eq5[1].set_color(RED)
        eq5[3].set_color(RED)

        eq6 = MathTex(r"f(x) = ", r"\beta_1", r"x + ", r"\beta_0").move_to((RIGHT + DOWN))
        eq6[1].set_color(RED)
        eq6[3].set_color(RED)

        eq7 = MathTex("f(x) = ", f'{m.get_value()}', r"x + ", f'{b.get_value()}').move_to((RIGHT + DOWN))
        eq7[1].set_color(RED)
        eq7[3].set_color(RED)

        # populate charting area
        self.play(DrawBorderThenFill(graph), run_time=2.0)

        # draw line
        self.play(Create(line), run_time=2.0)

        self.play(Create(eq5))

        note1 = Text("m: slope of the line",font_size=20).next_to(eq5, DOWN)
        note2 = Text("b: intercept of the line",font_size=20).next_to(note1, DOWN)

        # Hiển thị 2 note
        self.play(Write(note1))
        self.play(Write(note2))
        self.wait()

        # mô tả m và b trong biểu đồ
        def blink(item, value, increment):
            self.play(ScaleInPlace(item, 4 / 3), value.animate.increment_value(increment))
            # thay đổi tỉ lệ item thành 4/3 , tăng giá trị ( chữ to dần )
            for i in range(0, 1):
                self.play(ScaleInPlace(item, 3 / 4), value.animate.increment_value(-2 * increment))
                self.play(ScaleInPlace(item, 4 / 3), value.animate.increment_value(2 * increment))
            # tạo vòng lặp cho chữ nhỏ lại và to dần..
            self.play(ScaleInPlace(item, 3 / 4), value.animate.increment_value(-increment))
            self.wait()
        # chỉ chạy ở vị trí thứ 1 và 3
        blink(eq5[1], m, .50)
        blink(eq5[3], b, 2.0)

        self.wait()

        self.play(FadeOut(note1))
        self.play(FadeOut(note2))

        # transform to beta coefficients
        self.play(ReplacementTransform(eq5, eq6))
        self.wait()

        # transform with coefficient values
        self.play(ReplacementTransform(eq6, eq7))
        self.wait()

        # remove equation
        self.play(FadeOut(eq7, shift=DOWN))
        

def create_residual_model(scene,data,m,b,ax,points,line) -> tuple: # tạo phép tính phần thừa residuals
    residuals: list[Line] = [] #tạo list
    for d in data:
        residual = Line(start=ax.c2p(d.x, d.y),
                        end=ax.c2p(d.x, m.get_value() * d.x + b.get_value())).set_color(RED) #từ các điểm dot đến đường thẳng
        scene.play(Create(residual), run_time=.3)
        residual.add_updater( # cập nhật residuals khi m và b thay đổi
            lambda r,d=d:
                  r.become(Line(start=ax.c2p(d.x, d.y), end=ax.c2p(d.x, m.get_value()*d.x+b.get_value())).set_color(RED)))
        residuals += residual

  # flex residuals changing the coefficients m and b
    def flex_residuals():
        m_delta=1.1 #giá trị tăng giảm
        scene.play(m.animate.increment_value(m_delta)) #thay đổi giá trị m theo m_delta
        for i in (-1,1,-1,1): #tăng giảm giá trị m
            scene.play(m.animate.increment_value(i*m_delta))
            scene.play(m.animate.increment_value(i*m_delta))
        scene.play(m.animate.increment_value(-m_delta))
            
        scene.wait()

    return residuals, flex_residuals



class ThirdScene(Scene):

    def construct(self):
        # add base graph
        data, m, b, ax, points, line = create_model()
        self.add(ax, line, *points)

        residuals, flex_residuals = create_residual_model(self, data, m, b, ax, points, line)
        #vẽ hình vuông biểu diễn
        squares: list[Square] = []
        for i, d in enumerate(data):
            square = Square(color=RED,fill_opacity=.6,side_length=residuals[i].get_length()).next_to(residuals[i], LEFT, 0)

            square.add_updater(lambda s=square, r=residuals[i]: s.become(Square(color=RED,fill_opacity=.6,side_length=r.get_length()).next_to(r, LEFT, 0)))
            # residuals thay đổi thì hình vuông cũng thay đổi theo
            squares += square
            self.play(Create(square), run_time=.1)

        flex_residuals()
        length = 0.0
        #tính sse = tổng các diện tích hình vuông
        for s in squares:
            length = sqrt(length ** 2 + s.side_length ** 2)
            total_square = Square(color=RED, fill_opacity=1, side_length=length).move_to(3 * LEFT + 2.5 * UP) # hiện SSE lên trên hình vuông
            self.play(ReplacementTransform(s, total_square),run_time=.3)

        sse_text = Text("SSE", font_size=48).move_to(total_square)
        sse_label = Text("Sum Squared Error", font_size=48).to_corner(DR).shift(UP + LEFT).set_color(BLUE)
        self.play(DrawBorderThenFill(sse_text),Write(sse_label))

        # Adjust the position of the formula relative to total_square
        sse_formula = MathTex( "=", r"\sum_{i=1}^{n} (y_i - \hat{y}_i)^2").scale(1.5).next_to(total_square, RIGHT)
        self.play(Write(sse_formula))
        self.wait()
        label_y = MathTex(r"y_i: ",r"\text{Actual values}").to_corner(DR).shift(UP*3 + LEFT*1.5)
        label_y_hat = MathTex(r"\hat{y}_i: ", r"\text{Predicted values}").next_to(label_y, DOWN).align_to(label_y,LEFT)
        self.play(Create(label_y),Create(label_y_hat))
        self.wait(2)

        mse_formula = MathTex("=","\\frac{1}{n}", "\\sum_{i=1}^{n}", "(y_i - \\hat{y}_i)^2").scale(1.5).next_to(total_square, RIGHT)

        # Text labels
        mse_text = Text("MSE", font_size=40).move_to(total_square)
        mse_label = Text("Mean Squared Error", font_size=48).move_to(sse_label).set_color(BLUE)
        losf= Text("Loss function", font_size=48).move_to(sse_label).set_color(BLUE).align_to(sse_label,RIGHT)
        losf_formula = MathTex("=","\\frac{1}{2n}", "\\sum_{i=1}^{n}", "(y_i - \\hat{y}_i)^2").scale(1.5).next_to(total_square, RIGHT)
        losf_text = MathTex(r"J(\theta)", font_size=52).move_to(total_square)
        # Show the formula and labels
        self.play(ReplacementTransform(sse_text,mse_text),ReplacementTransform(sse_label,mse_label),ReplacementTransform(sse_formula,mse_formula))
        self.wait(2)
        self.play(ReplacementTransform(mse_text,losf_text),ReplacementTransform(mse_label,losf),ReplacementTransform(mse_formula,losf_formula))
        self.wait(2)

class OptimalModelScene(Scene):

    def construct(self):
        cont1 = Text("How to find the most optimal model?", font_size=40)
        cont2 = Text("Find m, b values to minimize Loss Function", font_size=40)
        cont3 = Text("Gradient Descent",font_size=48).set_color_by_gradient(RED,ORANGE)
        # Display the title
        self.play(Write(cont1), run_time=2)
        self.wait()

        # Fade out the title and display the subtitle
        self.play(FadeOut(cont1), run_time=1)
        self.play(Write(cont2), run_time=2)
        self.wait()
        self.play(FadeOut(cont2),run_time=1)
        self.wait()
        self.play(Write(cont3))
        self.wait()
#Ngọc 
class LinearRegressionLoss3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)
        #set góc cam cho hình 3D

        loss_axes = ThreeDAxes(x_range=(-10, 10, 1), y_range=(-10, 10, 1), z_range=(-1, 100_000, 10_000))
        #tạo trục 3D

        points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples()) #data set-> tuple -> list
        m, b, i, n = symbols('m b i n') # Symbols for slope, intercept, index, and the number of points
        x, y = symbols('x y', cls=Function) #Symbolic functions
#Tạo SS giữa các điểm với giá trị predict y(i), lặp từ 1 đến n
#thay n = len-1 -> tính sumsquare = doit
#thay x bằng x trong point chạy theo vòng
#thay y = y trong point chạy theo vòng

        _sum_of_squares = Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n)) \
            .subs(n, len(points) - 1).doit() \
            .replace(x, lambda i: points[i].x) \
            .replace(y, lambda i: points[i].y)
        
        sum_of_squares = lambdify([m, b], _sum_of_squares) #biến cái này thành hàm gọi được

        #Taọ tọa độ điểm trên 3d để phục vụ GD
        def param_loss(u, v):
            m = u#gắn các giá trị
            b = v
            y = sum_of_squares(m, b) #tạo giá trị ss với m, b tương ứng
            return loss_axes.c2p(m, b, y) #trả về
        # Create a surface representing the loss function
        loss_plane = Surface(
            param_loss,
            resolution=(42, 42),
            v_range=[-10, +10],
            u_range=[-10, +10]
        )
        loss_plane.set_style(fill_opacity=.5, stroke_color=BLUE)
        loss_plane.set_fill(BLUE)
        # Define the derivative of the sum of squares loss function with respect to m and b
        d_m = lambdify([m, b], diff(_sum_of_squares, m))
        d_b = lambdify([m, b], diff(_sum_of_squares, b))

        # Building the model
        m = -5.0
        b = 5.0
        # The learning Rate
        L = .0025

        # The number of iterations
        iterations = 1000

        path = VMobject(stroke_color=RED) #group các đường đỏ
        # Initialize a path to visualize the optimization trajectory
        dot = Sphere(radius=.05, stroke_color=YELLOW, fill_opacity=.2).move_to(param_loss(m, b))
        path.set_points_as_corners([dot.get_center(), dot.get_center()]) #set 1 điểm xuất phát, kết thúc
        #Cập nhật thêm điểm
        def update_path(path):
            previous_path = path.copy() #copy path đấy
            previous_path.add_points_as_corners([dot.get_center()]) #update
            path.become(previous_path) #đổi path
        #update ..
        path.add_updater(update_path)

        path_points = []

        # Perform Gradient Descent
        for i in range(iterations):
            # update m and b
            m -= d_m(m, b) * L
            b -= d_b(m, b) * L
            print(m, b)
            if round(b, 4) == 4.7333 and round(m, 4) == 1.9394:
                break

            path_points.append((m, b))
        # m= b=
        m_b_label = always_redraw(lambda: MathTex("m &= ", round(m, 4),
                                                  "\\\\b &= ", str(round(b, 4)),
                                                  font_size=40).move_to(LEFT * 5 + DOWN * 2))
        #create the graph in up left
        right_graph = VGroup(loss_plane, path, dot, loss_axes).move_to(RIGHT * 3.6 + UP).scale(.4)

        self.add(right_graph)
        self.add_fixed_in_frame_mobjects(m_b_label)

        # line graph
        linear_axes = Axes(x_range=(0, 10, -1), y_range=(0, 26, 2))
        points = [Dot(point=linear_axes.c2p(p.x, p.y), radius=.25, color=BLUE) for p in points]
        #line in 2d grph
        line = Line(start=linear_axes.c2p(b, 0), end=linear_axes.c2p(10, m * 10 + b))
        #group
        left_graph = VGroup(linear_axes, line, *points).move_to(LEFT * 3.3 + UP).scale(.4)
        #thêm nguyên nhóm kia vào biểu đồ
        self.add_fixed_in_frame_mobjects(left_graph)

        for i, p in enumerate(path_points):
            m = p[0]
            b = p[1]
        #chạy m, b là điểm xuất phát, chạy dot ở 3d để tạo path, chạy đường thẳng ở đồ thị bên trên
            self.play(
                dot.animate.move_to(param_loss(p[0], p[1])),
                Transform(line, Line(start=linear_axes.c2p(10, m * 10 + b), end=linear_axes.c2p(0, b))),
                run_time=max(.1, 1.0 - (i * .01))
            )

        self.wait(4)

def create_train_test_model() -> tuple:
    df = pd.read_csv('https://bit.ly/3TUCgh2', delimiter=",") #data set, cách bằng dấu , pandas đọc file

    X = df.values[:, :-1]#nhận từ đầu đến cột t2 từ cuối lên ( theo data là cột 1)
    Y = df.values[:, -1]#cột 2 cột cuối

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=7) #chia tập train test

    model = LinearRegression() #Chạy hồi quy tuyến tính từ scikit-
    fit = model.fit(X_train, Y_train) #lưu tập train vào biến fit
    result = model.score(X_test, Y_test) #chạy tập train -> hqtt

    m = ValueTracker(fit.coef_.flatten()[0]) #tạo tracker cho biến m theo mô hình
    b = ValueTracker(fit.intercept_.flatten()[0]) #tạo tracker cho biến b theo mô hình
#tạo trục x,y
    ax = Axes(
        x_range=[0, 100, 20],
        y_range=[-40, 200, 40],
        axis_config={"include_tip": False}, #cau hinh mac dinh tat
    )
    # tạo point trên trục xy
    train_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                    pd.DataFrame(data={'x': X_train.flatten(), 'y': Y_train.flatten()}).itertuples()]
    #tập traim mỳ xanh đường kính 0.15 trên trục xy
    #chạy vòng lặp tạo các tuple lặp theo 2 mảng 1 chiều x và y (flattern -> dữ liệu thành 1 mảng)
    test_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                   pd.DataFrame(data={'x': X_test.flatten(), 'y': Y_test.flatten()}).itertuples()]

    # plot function tạo đường thẳng tuyến tính, bắt đầu từ điểm (0,b) và điểm (200, giá trị của y=m*200+b) xét màu vàng
    line = Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value())).set_color(
        YELLOW)

    # make line follow m and b value. Sử dụng add_updater để cập nhận _ Lambda l -> line(..)
    line.add_updater(
        lambda l: l.become(
            Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value()))).set_color(
            YELLOW)
    )

    return m, b, ax, train_points, test_points, line #trả về m, b, trục, đường thẳng và 2 tập test, train


class TrainTestScene(Scene):
    
    def construct(self):
        #tạo chữ chuyển cảnh
        train_test_label= Text("Train/Test Process", font_size=48).set_color_by_gradient(BLUE,GREEN,BLUE)
        self.play(Write(train_test_label))
        self.wait(2)
        self.play(FadeOut(train_test_label))
        self.wait(2)
        m, b, ax, train_points, test_points, line = create_train_test_model() #hàm tạo model
        #Khởi tạo lại giá trị m b trục ngang và các data set vào model test tuyến tính
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)
        #Tạo 2 nhóm tập train vs test
        # populate charting area -> add 2 grouppoint vào cảnh
        self.add(ax, train_group, test_group)

        # Move train/test split to box visual box xanh
        training_rect = Rectangle(color=BLUE, height=1.5, width=1.5, fill_opacity=.8).shift(LEFT * 4 + UP * 2.5)
        training_text = Text("TRAIN").scale(.6).move_to(training_rect) #chữ train
        training_rect_grp = VGroup(training_rect, training_text) #nhóm chữ vs hôp lại thành 1 group
        #tương tự với hộp đỏ
        test_rect = Rectangle(color=RED, height=1.5 / 2, width=1.5, fill_opacity=.8).next_to(training_rect, DOWN)
        test_text = Text("TEST").scale(.6).move_to(test_rect)
        test_rect_grp = VGroup(test_rect, test_text)

        self.wait()
        self.play(*[p.animate.set_fill(RED) for p in test_points]) #tập point test thành màu đỏ
        self.wait()

        # copy training and testing points for reverse transformation
        train_points_copy = [p.copy() for p in train_points]
        test_points_copy = [p.copy() for p in test_points] #tạo 2 tập point

        self.play(ReplacementTransform(test_group, test_rect_grp)) # rồi dùng cái này tạo hiệu ứng từ chuyển point thành hộp
        self.wait()
        self.play(ReplacementTransform(train_group, training_rect_grp))

        b1 = Brace(test_rect, direction=RIGHT) #tạo dấu ngoặc nhọn phía bên phải hộp test
        b1_label = MathTex(r"\frac{1}{3}").scale(.7).next_to(b1, RIGHT) #tạo 1/3 bên phái ngoặc nhọn
        self.play(Create(b1), Create(b1_label)) # tạo

        b2 = Brace(training_rect, direction=RIGHT) # tương tự với dấu ngoặc ở tập train
        b2_label = MathTex(r"\frac{2}{3}").scale(.7).next_to(b2, RIGHT)
        self.play(Create(b2), Create(b2_label))

        self.wait()
        self.play(FadeOut(b1), FadeOut(b2), FadeOut(b1_label), FadeOut(b2_label)) #cho làm mờ ngoặc vs số
        self.wait()
        train_points_copy_grp = VGroup(*train_points_copy) #tạo group cho cacs tập copy
        test_points_copy_grp = VGroup(*test_points_copy)

        self.play(ReplacementTransform(training_rect_grp, train_points_copy_grp)) #Tạo hiệu ứng từ hình thành point train
        self.wait()
        self.play(Create(line)) #tạo đường thẳng
        self.wait()
        self.play(FadeOut(train_points_copy_grp)) #cho tập train mờ
        self.play(ReplacementTransform(test_rect_grp, test_points_copy_grp)) # cho tập test thành point
        self.wait()

class MetricScene(Scene):
    
    def construct(self):
        # Title
        title = Text("Performance Metrics", color=BLUE).shift(UP * 3 + LEFT * 3)
        self.play(Create(title))
        self.wait()

        # R² Score Explanation
        term0 = MathTex(r"\text{R}^2 \text{ Score}", color=YELLOW)
        term0desc = MathTex(r"\text{ - Scores the variation predictability of variables between 0 and 1.}").next_to(term0, RIGHT)
        term0grp = VGroup(term0, term0desc) \
            .scale(.70) \
            .next_to(title, DOWN, buff=1) \
            .align_to(title, LEFT)
        term00 = MathTex(r"\text{R}^2 \text{ Score} = 1-\frac{SSE}{SST}").next_to(term0, RIGHT, buff=6.5).scale(.70)
        term000 = MathTex(r"\text{R}^2 \text{ Score} = 1-\frac{\Sigma(y_i-\hat{y}_i)^2}{\Sigma(y_i-\overset{-}{y})^2}").move_to(term00).scale(.70)
        animations0 = [
            Create(term0grp),
            FadeOut(term0desc),
            Create(term00),
            ReplacementTransform(term00, term000),
        ]
        for animation0 in animations0:
            self.play(animation0)
            self.wait()

        # Standard Error of the Estimate Explanation
        term1 = MathTex(r"\text{Standard Error of the Estimate}", color=YELLOW)
        term1desc = MathTex(r"\text{ - Measures the overall error of the linear regression.}").next_to(term1, RIGHT)
        term1grp = VGroup(term1, term1desc) \
            .scale(.70) \
            .next_to(term0, DOWN, buff=1) \
            .align_to(title, LEFT)
        term01 = MathTex(r"SEE = \sqrt{\frac{\Sigma(x_i-\overset{-}{x})}{n-2}}").next_to(term1, RIGHT).scale(.70).align_to(term000,LEFT)
        animations1 = [
            Create(term1grp),
            FadeOut(term1desc),
            Create(term01),
        ]
        for animation1 in animations1:
            self.play(animation1)
            self.wait()

        # Mean Absolute Error Explanation
        term2 = MathTex(r"\text{Mean Absolute Error}", color=YELLOW)
        term2desc = MathTex(r"\text{ - Measures the overall absolute error of the linear regression.}").next_to(term2, RIGHT)
        term2grp = VGroup(term2, term2desc) \
            .scale(.70) \
            .next_to(term1, DOWN, buff=1) \
            .align_to(title, LEFT)
        term02 = MathTex(r"MAE = \frac{\Sigma\left|\hat{y}_1-y_i\right|}{n}").next_to(term2, RIGHT).scale(.70).align_to(term000,LEFT)
        animations2 = [
            Create(term2grp),
            FadeOut(term2desc),
            Create(term02),
        ]
        for animation2 in animations2:
            self.play(animation2)
            self.wait()

        # Mean Squared Error Explanation
        term3 = MathTex(r"\text{Mean Squared Error}", color=YELLOW)
        term3desc = MathTex(r"\text{ - Measures the average squared error of the linear regression.}").next_to(term3, RIGHT)
        term3grp = VGroup(term3, term3desc) \
            .scale(.70) \
            .next_to(term2, DOWN, buff=1) \
            .align_to(title, LEFT)
        term03 = MathTex(r"MSE = \frac{\Sigma(\hat{y}_i-y_i)^2}{n}").next_to(term3, RIGHT).scale(.70).align_to(term000,LEFT)
        animations3 = [
            Create(term3grp),
            FadeOut(term3desc),
            Create(term03),
        ]
        for animation3 in animations3:
            self.play(animation3)
            self.wait()

#Mây
def create_problem_model() -> tuple:
    # Read data from the provided CSV URL into a DataFrame
    df = pd.read_csv("https://bit.ly/45VEJfL", delimiter=",")

    # Extract features (X) and target variable (Y) from the DataFrame
    X = df.values[:, :-1]  # area
    Y = df.values[:, -1]  # price

    # Split the data into training and testing sets (ratio train, test 2: 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1 / 3,
                                                        random_state=49)

    # Create a DataFrame for the training data
    train_data = pd.DataFrame(data=X_train, columns=df.columns[:-1])
    train_data['y'] = Y_train
    train_data = list(train_data.itertuples(
    ))  # Convert DataFrame rows to namedtuples (để dễ xử lý data sau này)

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
        axis_config={
            "include_tip": True
        },
    ).add_coordinates()

    # Add axis labels to the coordinate system
    labels = ax.get_axis_labels(
        MathTex(r"\text{Size of House (m}^2\text{)}").scale(0.7),
        MathTex(r"\text{Price of House (million VND)}").scale(0.7))

    # Create dots representing training and testing data points on the coordinate system
    train_points = [
        Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE)
        for p in pd.DataFrame(data={
            'x': X_train.flatten(),
            'y': Y_train.flatten()
        }).itertuples()
    ]

    test_points = [
        Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE)
        for p in pd.DataFrame(data={
            'x': X_test.flatten(),
            'y': Y_test.flatten()
        }).itertuples()
    ]

    # Create a line representing the linear regression model (và cập nhật line với giá trị m,b -> m,b thay đổi thì line thay đổi theo)
    line = Line(start=ax.c2p(0, b.get_value()),
                end=ax.c2p(100,
                           m.get_value() * 100 +
                           b.get_value())).set_color(YELLOW)
    line.add_updater(lambda l: l.become(
        Line(start=ax.c2p(0, b.get_value()),
             end=ax.c2p(100,
                        m.get_value() * 100 + b.get_value()))).set_color(YELLOW
                                                                         ))
    # Return the generated elements as a tuple
    return train_data, m, b, ax, train_points, test_points, line, labels, result


def sse_fit(scene, data, m, b, ax, points, line) -> tuple:
    # List to store residuals (lines representing the vertical distance between data points and the regression line)
    residuals: list[Line] = []

    # Iterate through each data point (data truyền vào ban đầu là train set)
    for d in data:
        # Create a residual line for the current data point and animate its appearance
        residual = Line(start=ax.c2p(d.x, d.y),
                        end=ax.c2p(d.x,
                                   m.get_value() * d.x +
                                   b.get_value())).set_color(RED)
        scene.play(Create(residual), run_time=.2)

        # Add an updater to dynamically adjust the residual line during animation
        residual.add_updater(lambda r, d=d: r.become(
            Line(start=ax.c2p(d.x, d.y),
                 end=ax.c2p(d.x,
                            m.get_value() * d.x + b.get_value())).set_color(RED
                                                                            )))
        residuals += residual  # Add the residual line to the list

    # Function to calculate the Sum of Squared Errors (SSE) given current values of slope (m) and intercept (b)
    def get_sse(m, b):
        new_sse = ValueTracker(0.0)
        for i in range(len(residuals)):
            new_sse += (residuals[i].get_length()**2)
        return new_sse

    # Return the list of residuals and the SSE calculation function
    return residuals, get_sse


class TableExamples(Scene):
    #Cảnh mở đầu ví dụ(câu hỏi và bảng)
    def construct(self):
        # Create title and subtitle text objects
        title = Text("Linear Regression")
        subTitle = Text("Hanoi Housing Price Problems").set_color_by_gradient(
            BLUE, GREEN)

        # Create problem statement text objects
        problem = Text(
            "Given the historical house selling prices in the table,",
            font_size=32).to_edge(UP).set_color(YELLOW)
        problem2 = Text("can you find the optimal price of a 70 m² house?",
                        font_size=32).next_to(problem, DOWN,
                                              buff=0.2).set_color(YELLOW)

        # Adjust title and subtitle positions
        title.shift(0.5 * UP)
        subTitle.next_to(title, DOWN)

        # Animate the appearance and movement of title and subtitle
        self.play(Write(title), Write(subTitle))
        self.play(ApplyMethod(title.shift, 3 * UP), FadeOut(subTitle),
                  FadeOut(title))

        # Animate the appearance of problem statements with a lagged start
        self.play(
            LaggedStart(Write(problem),
                        Write(problem2),
                        lag_ratio=0.4,
                        run_time=3))

        # Create a historical data table
        t2 = Table([["30", "48"], ["40", "62"], ["50", "75"]],
                   col_labels=[Text("Area"), Text("Price")],
                   include_outer_lines=True)

        # Set colors for column labels in the historical data table
        lab_2 = t2.get_col_labels()
        for item in lab_2:
            item.set_color(YELLOW)

        # Create outline for the historical data table and scale it
        outline_2 = VGroup(t2.vertical_lines, t2.horizontal_lines)
        t2.scale(0.5).shift(LEFT * 4)

        # Create a prediction data table
        t1 = Table([["30", "48"], ["40", "62"], ["50", "75"], ["70", "?"],
                    ["120", "?"], ["150", "?"]],
                   col_labels=[Text("Area"), Text("Price")],
                   include_outer_lines=True)

        # Set colors for column labels in the prediction data table
        lab_1 = t1.get_col_labels()
        for item in lab_1:
            item.set_color(YELLOW)

        # Create outline for the prediction data table and scale it
        outline_1 = VGroup(t1.vertical_lines, t1.horizontal_lines)
        t1.scale(0.5).shift(RIGHT * 4)

        # Get rows from the prediction data table
        row_1 = t1.get_rows()
        row1to3 = VGroup(row_1[1], row_1[2], row_1[3])
        rowlast = VGroup(row_1[4], row_1[5], row_1[6])

        # Set color for the last row in the prediction data table
        rowlast.set_color(BLUE)

        # Define mathematical equations for linear regression
        eq1 = MathTex("Area", "=?.Price+?").next_to(t1, DOWN, buff=1)
        eq2 = MathTex(r"y", r"= \beta_1", "x", r"+ \beta_0").next_to(t1,
                                                                     DOWN,
                                                                     buff=1)
        eq3 = MathTex("y", "=", "mx", "+", "b").next_to(t1, DOWN, buff=1)

        # Create a framebox around the first equation
        framebox1 = SurroundingRectangle(eq1)

        # Create arrows to represent the flow of information
        arrow_1 = Arrow(start=t2.get_right(), end=t1.get_left()).set_angle(0.2)
        arrow_2 = Arrow(start=t2.get_right(), end=t1.get_left(),
                        color=BLUE).set_angle(-0.2)
        arrow_3 = Arrow(start=t1.get_bottom(),
                        end=framebox1.get_top(),
                        color=YELLOW).scale(1.5)

        # Create braces with accompanying text for historical data and prediction
        b1 = Brace(arrow_1,
                   direction=arrow_1.copy().rotate(PI / 2).get_unit_vector())
        b1text = b1.get_tex(r"\text{Historical data}").scale(0.7)
        b2 = Brace(arrow_2,
                   direction=arrow_2.copy().rotate(-PI / 2).get_unit_vector(),
                   color=BLUE)
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
        # Create a linear regression problem and obtain necessary elements từ hàm đã tạo
        train_data, m, b, ax, train_points, test_points, line_f, labels, result = create_problem_model(
        )

        ## Show dataset (both data + train)
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)
        self.play(Create(ax), Write(labels))
        self.play(Create(train_group), Create(test_group))
        self.wait()

        # Split the dataset into training and testing sets
        self.play(*[p.animate.set_fill(RED) for p in test_points])
        self.wait()
        self.play(FadeOut(test_group))
        self.wait()

        # Visualize the initial linear regression model (bad fit)
        m_1 = ValueTracker(0)
        b_1 = ValueTracker(100)
        SSE_of_best_func = ValueTracker(12.781)
        eq1 = always_redraw(lambda: MathTex(f"y = {m_1.get_value():.3f}",
                                            r"x + ", f"{b_1.get_value():.3f}").
                            move_to((LEFT * 3 + UP * 2.5)))
        line_1 = Line(start=ax.c2p(0, b_1.get_value()),
                      end=ax.c2p(100,
                                 m_1.get_value() * 100 +
                                 b_1.get_value())).set_color(YELLOW)
        line_1.add_updater(lambda l: l.become(
            Line(start=ax.c2p(0, b_1.get_value()),
                 end=ax.c2p(100,
                            m_1.get_value() * 100 + b_1.get_value()))).
                           set_color(YELLOW))
        sse_text_1 = MathTex("SSE = ").next_to(eq1, DOWN, buff=0.3)

        # Animate the creation of the initial model (best) and SSE visualization
        self.play(Create(line_1))
        self.play(Create(eq1), Write(sse_text_1))
        residuals_b, get_sse = sse_fit(self, train_data, m_1, b_1, ax,
                                       train_points, line_1)
        sse_1 = get_sse(m_1, b_1)
        sse_text_2 = always_redraw(lambda: MathTex(
            "SSE = ", f'{sse_1.get_value():.3f}').next_to(eq1, DOWN, buff=0.3))
        self.play(ReplacementTransform(sse_text_1, sse_text_2))
        self.wait(2)

        # Optimize the model to the best fit. Khi line chuyển từ các hệ số của bad fit -> best fit.
        self.play(m_1.animate.set_value(m.get_value()),
                  b_1.animate.set_value(b.get_value()),
                  sse_1.animate.set_value(SSE_of_best_func.get_value()),
                  run_time=3,
                  rate_func=linear)
        self.wait(2)


class FinalExample2(Scene):
    
    def construct(self):
        # Create a linear regression problem and obtain necessary elements
        data, m, b, ax, train_points, test_points, line_f, labels, result = create_problem_model(
        )
        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)

        # Display the training and testing data points, along with the initial linear regression model
        eq1 = MathTex(f"y = {m.get_value():.3f}", r"x + ",
                      f"{b.get_value():.3f}").move_to(LEFT * 2.5 + UP * 2.3)
        r2_score = MathTex("R^2 \\text{ Score: }",
                           f"{result.get_value():.3f}").next_to(
                               eq1, DOWN, buff=0.5).align_to(eq1, LEFT)
        r2_score[1].set_color(RED)
        test_group.set_color(RED)

        # Prediction visualization for a 70 m² house
        point = ax.coords_to_point(70, m.get_value() * 70 + b.get_value())
        u_point = ax.c2p(0, m.get_value() * 70 + b.get_value())
        price_y = MathTex(f"{m.get_value()*70+b.get_value():.0f}").next_to(
            u_point, LEFT, buff=0.3).set_color(RED)
        dot = Dot(point)
        ver_line = ax.get_vertical_line(point,
                                        line_config={"dashed_ratio": 0.85})
        hori_line = ax.get_horizontal_line(point,
                                           line_config={"dashed_ratio": 0.85})

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
        left_panel = VGroup(ax, train_group, labels, eq1, line_f, test_group,
                            r2_score, hori_line, dot, ver_line,
                            price_y).scale_to_fit_height(5)

        # Display conclusion text
        conclusion_text = Text(
            "Conclusion: The optimal price of a 70 m² house in Hanoi,").scale(
                0.5).to_edge(UP)
        con2 = Text("based on our Linear Regression model is 127 million VND."
                    ).scale(0.5).next_to(conclusion_text, DOWN, buff=0.3)
        self.play(left_panel.animate.to_edge(LEFT).to_edge(DOWN), run_time=4)
        self.play(Write(conclusion_text))
        self.play(Write(con2))
        self.wait()

class TextScene(Scene):
    
    def construct(self):
        credit_text1 = Text("GROUP 6",
                            weight=BOLD,
                            font="Open Sans",
                            font_size=32)
        credit_text2 = Text("DIRECTED BY",
                            weight=LIGHT,
                            font="Open Sans",
                            font_size=16).next_to(credit_text1, UP, buff=0.1)

        self.play(Create(credit_text1), Create(credit_text2))
        self.wait(2)
        self.play(FadeOut(credit_text1), FadeOut(credit_text2))

        # Group 6
        group_text = [
            "Nguyễn Phương Hoài Ngọc",
            "Võ Huyền Khánh Mây",
            "Võ Thị Minh Phương",
            "Dương Nhật Thanh",
            "Trần Phương Linh",
        ]
        text_group = VGroup(*[
            Text(text, weight=BOLD, font="Open Sans", font_size=32).scale(0.7)
            for text in group_text
        ])

        # Set the initial position of the group at the bottom of the screen
        text_group.arrange(DOWN).move_to(5.5 * DOWN)

        # Play the animation to write and shift the group of texts
        self.play(Create(text_group), run_time=0.1)
        self.play(text_group.animate.shift(12 * UP), run_time=8)
        # Thanks for watching
        thanks_text = Text("Thanks for watching",
                           weight=BOLD,
                           font="Open Sans",
                           font_size=32).move_to(5 * DOWN)
        self.play(Create(thanks_text), run_time=0.2)
        self.play(thanks_text.animate.shift(5 * UP), run_time=3)


# In[ ]:




