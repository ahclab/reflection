import torch
import torch.nn as nn
import torch.nn.functional as F
    

def reflection(x, a, c):
    '''
        Args:
            x: the word vector
            a: the vector orthogonal to the mirror
            c: the vector on the mirror
        Return:
            the output of reflection. y = Ref_a,c(x)
    '''
    assert x.size()==a.size()==c.size(), \
    'x.size()={}, a.size()={}, c.size()={}'.format(x.size(), a.size(), c.size())
    dot_aa = torch.matmul(a, torch.t(a))
    dot_aa = torch.diag(dot_aa, 0)
    dot_aa = torch.reshape(dot_aa, (len(dot_aa),1))
    dot_xca = torch.matmul(x-c, torch.t(a))
    dot_xca = torch.diag(dot_xca, 0)
    dot_xca = torch.reshape(dot_xca, (len(dot_xca),1))
    return x - 2 * (dot_xca / dot_aa) * a


class Ref_PM(nn.Module):
    def __init__(self, dim_x, dim_h, n_attributes=4):
        super(Ref_PM, self).__init__()
        self.embed_z = nn.Embedding(n_attributes, dim_x)
        
        self.wa1 = nn.Linear(dim_x, dim_h)
        self.wa2 = nn.Linear(dim_x, dim_h)
        self.wa3 = nn.Linear(dim_h, dim_h)
        self.wa4 = nn.Linear(dim_h, dim_h)
        self.wa5 = nn.Linear(dim_h, dim_h)
        self.wa6 = nn.Linear(dim_h, dim_h)
        self.wa7 = nn.Linear(dim_h, dim_x)
        
        self.wc1 = nn.Linear(dim_x, dim_h)
        self.wc2 = nn.Linear(dim_x, dim_h)
        self.wc3 = nn.Linear(dim_h, dim_h)
        self.wc4 = nn.Linear(dim_h, dim_h)
        self.wc5 = nn.Linear(dim_h, dim_h)
        self.wc6 = nn.Linear(dim_h, dim_h)
        self.wc7 = nn.Linear(dim_h, dim_x)
        
    def mlp_a(self, z, x):
        a = self.wa1(z) + self.wa2(x)
        a = self.wa3(F.relu(a))
        a = self.wa4(F.relu(a))
        a = self.wa5(F.relu(a))
        a = self.wa6(F.relu(a))
        a = self.wa7(F.relu(a))
        return a
        
    def mlp_c(self, z, x):
        c = self.wc1(z) + self.wc2(x)
        c = self.wc3(F.relu(c))
        c = self.wc4(F.relu(c))
        c = self.wc5(F.relu(c))
        c = self.wc6(F.relu(c))
        c = self.wc7(F.relu(c))
        return c
    
    def forward(self, x, z):   
        '''
        Args
           x: the input word vector
           z: the attribute ID (e.g. 0)
        Return:
           y: the output vector
        '''        
        # Embed z
        z = self.embed_z(z)
        
        # Estimate a and c with parameterized mirrors
        a = self.mlp_a(z, x)
        c = self.mlp_c(z, x)
        
        # Transfer the word vector with reflection
        return reflection(x, a, c) 
    
    
class Ref_PM_Share(nn.Module):
    def __init__(self, dim_x, dim_h, n_attributes=4):
        super(Ref_PM_Share, self).__init__()
        self.embed_z = nn.Embedding(n_attributes, dim_x)
        self.w1 = nn.Linear(dim_x, dim_h)
        self.w2 = nn.Linear(dim_x, dim_h)
        self.w3 = nn.Linear(dim_h, dim_h)
        self.w4 = nn.Linear(dim_h, dim_h)
        self.w5 = nn.Linear(dim_h, dim_h)
        self.w6 = nn.Linear(dim_h, dim_h)
        self.wa = nn.Linear(dim_h, dim_x)
        self.wc = nn.Linear(dim_h, dim_x)

    def mlp_ac(self, z, x):
        o = self.w1(z) + self.w2(x)
        o = self.w3(F.relu(o))
        o = self.w4(F.relu(o))
        o = self.w5(F.relu(o))
        o = self.w6(F.relu(o))
        a = self.wa(F.relu(o))
        c = self.wc(F.relu(o))
        return a, c
        
    def forward(self, x, z):   
        '''
        Args
           x: the input word vector
           z: the attribute ID (e.g. 0)
        Return:
           y: the output vector
        '''        
        # Embed z
        z = self.embed_z(z)
        
        # Estimate a and c with parameterized mirrors
        a, c = self.mlp_ac(z, x)
        
        # Transfer the word vector with reflection
        return reflection(x, a, c) 