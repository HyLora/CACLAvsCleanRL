import numpy
def map_output_color(output,failure_message="No Visuals for outputs _____(add failure_message)") -> dict: 
    min = numpy.array(numpy.amin(output))
    max = numpy.array(numpy.amax(output))
    rgb = [[ (0,0,0) for _ in column] for column in output]

    if max - min != 0.0:
        for x,column in enumerate(output):
            for y,_ in enumerate(column):
                color_val = 0
                try:
                    color_val = int(255.0 * (output[x][y] -min) / (max - min))
                except:
                    pass
                # TODO for some reason the mapping is sometimes of (maybe precision of float?)
                if color_val > 255:
                    color_val = 255 
                if color_val < 0: 
                    color_val = 0
                rgb[x][y] = (int(color_val),0,0) 
        rgb = numpy.array(rgb)
    return {"max":max,"min":min,"colors":rgb}

def line_intersect(A1, A2 ,B1 ,B2 ):
    """ returns a (x, y) tuple or None if there is no intersection """
    Ax1, Ay1 = A1
    Ax2, Ay2 = A2
    Bx1, By1 = B1
    Bx2, By2 = B2
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return None
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y

def is_in_rect(point,rect_point_0,rect_point_1):
    """returns if point is in the rect spanned by rect_point_0 and rect_point1"""
    if (rect_point_0[0] <= point[0] <= rect_point_1[0] or
        rect_point_0[0] >= point[0] >= rect_point_1[0]):
        if(rect_point_0[1] <= point[1] <= rect_point_1[1] or
            rect_point_0[1] >= point[1] >= rect_point_1[1]):
            return True
    return False

