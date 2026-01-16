import numpy as np

def get_alfa_conv(grad1=None, grad2=None, grad3=None, grad4=None):

    if grad4 is not None:
        g1 = grad1.reshape(-1)
        g2 = grad2.reshape(-1)
        g3 = grad3.reshape(-1)
        g4 = grad4.reshape(-1)
    
        g = np.concatenate((g1, g2, g3, g4)).astype(np.float32)
        alfa = .05 / np.max(g)
        
        # print(f'alfa conv: {alfa}')
        return alfa      

    else:
        g1 = grad1.reshape(-1)
        g2 = grad2.reshape(-1)
        g3 = grad3.reshape(-1)
    
        g = np.concatenate((g1, g2, g3)).astype(np.float32)
        
        alfa = .05 / np.max(g)

        # print(f'alfa conv: {alfa}')
        return alfa
        

def get_alfa_g(yk1=None, yk2=None, yk3=None, yk4=None, sk1=None, sk2=None, sk3=None, sk4=None):

    if yk4 is not None:
        m1 = sk1.reshape(-1)
        m2 = sk2.reshape(-1)
        m3 = sk3.reshape(-1)
        m4 = sk4.reshape(-1)
        
        m = np.concatenate((m1, m2, m3, m4)).astype(np.float32)
        
        term1 = np.dot(m, m)   
        
        g1 = yk1.reshape(-1)
        g2 = yk2.reshape(-1)
        g3 = yk3.reshape(-1)
        g4 = yk4.reshape(-1)
        
        g = np.concatenate((g1, g2, g3, g4)).astype(np.float32)
    
        term2 = np.dot(m, g)
        term3 = np.dot(g, g)
        
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
            
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1
            
        # print(f'alfa g: {alfa}')
        return alfa

    else:
        m1 = sk1.reshape(-1)
        m2 = sk2.reshape(-1)
        m3 = sk3.reshape(-1)
        
        m = np.concatenate((m1, m2, m3)).astype(np.float32)
        
        term1 = np.dot(m, m)   
        
        g1 = yk1.reshape(-1)
        g2 = yk2.reshape(-1)
        g3 = yk3.reshape(-1)
        
        g = np.concatenate((g1, g2, g3)).astype(np.float32)
    
        term2 = np.dot(m, g)
        term3 = np.dot(g, g)
        
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
            
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1

        # print(f'alfa g: {alfa}')
        return alfa 

# def get_alfa_conv(grad1=None, grad2=None, grad3=None, grad4=None,
#                   base_step=0.05, eps=1e-12):
    
#     if grad4 is not None:
#         g1 = grad1.reshape(-1)
#         g2 = grad2.reshape(-1)
#         g3 = grad3.reshape(-1)
#         g4 = grad4.reshape(-1)
#         g = np.concatenate((g1, g2, g3, g4)).astype(np.float32)
#     else:
#         g1 = grad1.reshape(-1)
#         g2 = grad2.reshape(-1)
#         g3 = grad3.reshape(-1)
#         g = np.concatenate((g1, g2, g3)).astype(np.float32)

#     g_abs_max = float(np.max(np.abs(g)))

#     if (not np.isfinite(g_abs_max)) or (g_abs_max < eps):
#         return 1e-3

#     alfa = base_step / g_abs_max
#     if (not np.isfinite(alfa)) or alfa <= 0:
#         return 1e-3

#     return float(alfa)


# def get_alfa_g(yk1=None, yk2=None, yk3=None, yk4=None,
#                sk1=None, sk2=None, sk3=None, sk4=None,
#                eps=1e-12):
    
#     if yk4 is not None:
#         m1 = sk1.reshape(-1)
#         m2 = sk2.reshape(-1)
#         m3 = sk3.reshape(-1)
#         m4 = sk4.reshape(-1)
#         m = np.concatenate((m1, m2, m3, m4)).astype(np.float32)

#         g1 = yk1.reshape(-1)
#         g2 = yk2.reshape(-1)
#         g3 = yk3.reshape(-1)
#         g4 = yk4.reshape(-1)
#         g = np.concatenate((g1, g2, g3, g4)).astype(np.float32)
#     else:
#         m1 = sk1.reshape(-1)
#         m2 = sk2.reshape(-1)
#         m3 = sk3.reshape(-1)
#         m = np.concatenate((m1, m2, m3)).astype(np.float32)

#         g1 = yk1.reshape(-1)
#         g2 = yk2.reshape(-1)
#         g3 = yk3.reshape(-1)
#         g = np.concatenate((g1, g2, g3)).astype(np.float32)

#     term1 = float(np.dot(m, m))
#     term2 = float(np.dot(m, g))
#     term3 = float(np.dot(g, g))

#     if (not np.isfinite(term1) or
#         not np.isfinite(term2) or
#         not np.isfinite(term3)):
#         return np.nan

#     if abs(term2) < eps or abs(term3) < eps:
#         return np.nan

#     abb1 = term1 / term2
#     abb2 = term2 / term3

#     if (not np.isfinite(abb1)) or (not np.isfinite(abb2)) or abs(abb1) < eps:
#         return 1e-3 #np.nan

#     abb3 = abb2 / abb1

#     if 0 < abb3 < 1:
#         alfa = abb2
#     else:
#         alfa = abb1

#     if (not np.isfinite(alfa)) or alfa <= 0:
#         return 1e-3 #np.nan

#     return float(alfa)
