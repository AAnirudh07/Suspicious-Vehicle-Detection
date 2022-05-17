Import math
      # Get center point of new object
      for rect in objects_rect:
          x, y, w, h, index = rect
          cx = (x + x + w) // 2
          cy = (y + y + h) // 2
          # Find out if that object was detected already
          same_object_detected = False
          for id, pt in self.center_points.items():
              dist = math.hypot(cx - pt[0], cy - pt[1])
              if dist < 25:
                  self.center_points[id] = (cx, cy)
                  # print(self.center_points)
                  objects_bbs_ids.append([x, y, w, h, id, index])
                  same_object_detected = True
                  break