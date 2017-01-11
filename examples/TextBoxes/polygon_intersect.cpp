#include "mex.h"
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

using namespace std;

typedef struct Point
{
    double x;//int
    double y;//int
}Point;

//########### Function: GetCrossPoint ###########
bool IsRectCross(const Point &p1,const Point &p2,const Point &q1,const Point &q2)
{
    bool ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
                min(q1.x,q2.x) <= max(p1.x,p2.x) &&
                min(p1.y,p2.y) <= max(q1.y,q2.y) &&
                min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}
bool IsLineSegmentCross(const Point &pFirst1,const Point &pFirst2,const Point &pSecond1,const Point &pSecond2)
{	
    long line1,line2;
    line1 = pFirst1.x * (pSecond1.y - pFirst2.y) +
        pFirst2.x * (pFirst1.y - pSecond1.y) +
        pSecond1.x * (pFirst2.y - pFirst1.y);
    line2 = pFirst1.x * (pSecond2.y - pFirst2.y) +
        pFirst2.x * (pFirst1.y - pSecond2.y) + 
        pSecond2.x * (pFirst2.y - pFirst1.y);
    if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
        return false;

    line1 = pSecond1.x * (pFirst1.y - pSecond2.y) +
        pSecond2.x * (pSecond1.y - pFirst1.y) +
        pFirst1.x * (pSecond2.y - pSecond1.y);
    line2 = pSecond1.x * (pFirst2.y - pSecond2.y) + 
        pSecond2.x * (pSecond1.y - pFirst2.y) +
        pFirst2.x * (pSecond2.y - pSecond1.y);
    if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
        return false;
    return true;
}

bool GetCrossPoint(const Point &p1,const Point &p2,const Point &q1,const Point &q2,long &x,long &y)
{
    if(IsRectCross(p1,p2,q1,q2))
    {
		if((q1.x-q2.x)==0){
			if((p1.x==q1.x) && (p1.y<=max(q1.y,q2.y)) && (p1.y>=min(q1.y,q2.y))){
				x=p1.x;
				y=p1.y;
				return true;
			}
			if((p2.x==q1.x) && (p2.y<=max(q1.y,q2.y)) && (p2.y >= min(q1.y,q2.y))){
				x=p2.x;
				y=p2.y;
				return true;
			}		
		}
		if((q1.y-q2.y)==0){
			if((p1.y==q1.y) && (p1.x<=max(q1.x,q2.x)) && (p1.x>=min(q1.x,q2.x))){
				x=p1.x;
				y=p1.y;
				return true;
			}
			if((p2.y==q1.y) && (p2.x<=max(q1.x,q2.x)) && (p2.x >= min(q1.x,q2.x))){
				x=p2.x;
				y=p2.y;
				return true;
			}		
		}
		if(((q1.x-q2.x)!=0) &&((q1.y-q2.y)!=0)){
			double delta_x,delta_y;
			delta_y = q1.y - q2.y;
			delta_x = q1.x - q2.x;
			//discrimination p1
			if(double(p1.y-q1.y)/delta_y == double(p1.x-q1.x)/delta_x &&(p1.x<=max(q1.x,q2.x)) && (p1.x>=min(q1.x,q2.x))){
				x=p1.x;
				y=p1.y;
				return true;
			}
			//discrimination p2
			if(double(p2.y-q1.y)/delta_y == double(p2.x-q1.x)/delta_x && (p2.x<=max(q1.x,q2.x)) && (p2.x >= min(q1.x,q2.x))){
				x=p2.x;
				y=p2.y;
				return true;
			}
		}
		if((p1.x-p2.x)==0){
			if((q1.x==p1.x) && (q1.y<=max(p1.y,p2.y)) && (q1.y>=min(p1.y,p2.y))){
				x=q1.x;
				y=q1.y;
				return true;
			}
			if((q2.x==p1.x) && (q2.y<=max(p1.y,p2.y)) && (q2.y >= min(p1.y,p2.y))){
				x=q2.x;
				y=q2.y;
				return true;
			}		
		}
		if((p1.y-p2.y)==0){
			if((q1.y==p1.y) && (q1.x<=max(p1.x,p2.x)) && (q1.x>=min(p1.x,p2.x))){
				x=q1.x;
				y=q1.y;
				return true;
			}
			if((q2.y==p1.y) && (q2.x<=max(p1.x,p2.x)) && (q2.x >= min(p1.x,p2.x))){
				x=q2.x;
				y=q2.y;
				return true;
			}		
		}
		if(((p1.x-p2.x)!=0) &&((p1.y-p2.y)!=0)){
			double delta_x,delta_y;
			delta_y = p1.y - p2.y;
			delta_x = p1.x - p2.x;
			//discrimination p1
			if(double(q1.y-p1.y)/delta_y == double(q1.x-p1.x)/delta_x && (q1.x<=max(p1.x,p2.x)) && (q1.x>=min(p1.x,p2.x))){
				x=q1.x;
				y=q1.y;
				return true;
			}
			//discrimination p2
			if(double(q2.y-p1.y)/delta_y == double(q2.x-p1.x)/delta_x && (q2.x<=max(p1.x,p2.x)) && (q2.x >= min(p1.x,p2.x))){
				x=q2.x;
				y=q2.y;
				return true;
			}
		}
		////////////////////////////////////////////////////////

        if (IsLineSegmentCross(p1,p2,q1,q2))
        {
			if((((p1.x-p2.x)==0) && ((q1.x-q2.x)==0)) || (((p1.y-p2.y)==0)&&((q1.y-q2.y)==0))){
				return false;
			}
			if(((p1.x-p2.x)!=0)&&((q1.x-q2.x)!=0)){
				double k1,k2;
				k1=fabs(double(p1.y-p2.y)/double(p1.x-p2.x));
				k2=fabs(double(q1.y-q2.y)/double(q1.x-q2.x));
				if(k1==k2){
					return false;
				}
			}
			
            long tmpLeft,tmpRight;
            tmpLeft = (q2.x - q1.x) * (p1.y - p2.y) - (p2.x - p1.x) * (q1.y - q2.y);
            tmpRight = (p1.y - q1.y) * (p2.x - p1.x) * (q2.x - q1.x) + q1.x * (q2.y - q1.y) * (p2.x - p1.x) - p1.x * (p2.y - p1.y) * (q2.x - q1.x);

            x = (int)((double)tmpRight/(double)tmpLeft);

            tmpLeft = (p1.x - p2.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q1.x - q2.x);
            tmpRight = p2.y * (p1.x - p2.x) * (q2.y - q1.y) + (q2.x- p2.x) * (q2.y - q1.y) * (p1.y - p2.y) - q2.y * (q1.x - q2.x) * (p2.y - p1.y); 
            y = (int)((double)tmpRight/(double)tmpLeft);
            return true;
        }
    }
    return false;
}
//################ Function: IsPointInPolygon ##################
//  The function will return YES if the point x,y is inside the polygon, or
//  NO if it is not.  If the point is exactly on the edge of the polygon,
//  then the function may return YES or NO.
bool IsPointInPolygon(std::vector<Point> poly, Point pt)
{
	int i,j;
    bool c = false;
    for (i = 0,j = poly.size() - 1;i < poly.size();j = i++)
    {
        if ((((poly[i].y <= pt.y) && (pt.y < poly[j].y)) ||
            ((poly[j].y <= pt.y) && (pt.y < poly[i].y)))
            && (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y)/(poly[j].y - poly[i].y) + poly[i].x))
        {
            c = !c;
        }
    }
    return c;
}

//############### Function: ClockwiseSortPoints ################
bool PointCmp(const Point &a,const Point &b,const Point &center)
{
    if (a.x >= 0 && b.x < 0)
        return true;
    if (a.x == 0 && b.x == 0)
        return a.y > b.y;
    int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
    if (det < 0)
        return true;
    if (det > 0)
        return false;
    int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
    int d2 = (b.x - center.x) * (b.x - center.y) + (b.y - center.y) * (b.y - center.y);
    return d1 > d2;
}
void ClockwiseSortPoints(std::vector<Point> &vPoints)
{
    //º∆À„÷ÿ–ƒ
    Point center;
    double x = 0,y = 0;
    for (int i = 0;i < vPoints.size();i++)
    {
        x += vPoints[i].x;
        y += vPoints[i].y;
    }
    center.x = (int)x/vPoints.size();
    center.y = (int)y/vPoints.size();

    //√∞≈›≈≈–Ú
    for(int i = 0;i < vPoints.size() - 1;i++)
    {
        for (int j = 0;j < vPoints.size() - i - 1;j++)
        {
            if (PointCmp(vPoints[j],vPoints[j+1],center))
            {
                Point tmp = vPoints[j];
                vPoints[j] = vPoints[j + 1];
                vPoints[j + 1] = tmp;
            }
        }
    }
}


//########### Function: PolygonClip ############
bool comparing(Point &p1,Point &p2)
{
	//sorted by comparing x
	if(p1.x < p2.x){
		return true;	
	}
	else{
		return false;	
	}
}
bool PolygonClip(const vector<Point> &poly1,const vector<Point> &poly2, std::vector<Point> &interPoly)
{
    if (poly1.size() < 3 || poly2.size() < 3)
    {
        return false;
    }

    long x,y;
    for (int i = 0;i < poly1.size();i++)
    {
        int poly1_next_idx = (i + 1) % poly1.size();
        for (int j = 0;j < poly2.size();j++)
        {
            int poly2_next_idx = (j + 1) % poly2.size();
            if (GetCrossPoint(poly1[i],poly1[poly1_next_idx],
                poly2[j],poly2[poly2_next_idx],
                x,y))
            {
				Point p;
				p.x = x;
				p.y = y;
                interPoly.push_back(p);
            }
        }
    }

    for(int i = 0;i < poly1.size();i++)
    {
        if (IsPointInPolygon(poly2,poly1[i]))
        {
            interPoly.push_back(poly1[i]);
        }
    }
    for (int i = 0;i < poly2.size();i++)
    {
        if (IsPointInPolygon(poly1,poly2[i]))
        {
            interPoly.push_back(poly2[i]);
        }
    }
    
    if(interPoly.size() <= 0)
        return false;
    //sort
	
	sort(interPoly.begin(),interPoly.end(),comparing);
	//vector<Point>::iterator iter=unique(interPoly.begin(),interPoly.end()); 
	int nPoly = interPoly.size();
	vector<int> flags;
	for(int ii=0;ii<nPoly;ii++)
	{
		int flag=1;
		flags.push_back(flag);
	}
	for(int ii=nPoly-1;ii>0;ii--)
	{
		int flag=0;
		for(int jj=0;jj<ii;jj++){
			if((interPoly[ii].x == interPoly[jj].x) && (interPoly[ii].y == interPoly[jj].y))
			{
				flag=1;
				break;
			}

		}
		if(flag==1){
			flags[ii]=0;
		}
	}
	vector<Point> tmp;
	for(int ii=0;ii<nPoly;ii++){
		if(flags[ii]==1){
			tmp.push_back(interPoly[ii]);
		}
	}
	interPoly.clear();
	interPoly = tmp;

    ClockwiseSortPoints(interPoly);
    return true;
}

//float int_area=polygon_intersect(array x1,array y1,array x2,array y2)
//eg: int_area = polygon_intersect(x1,y1,x2,y2);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *int_area;
    plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
    int_area=mxGetPr(plhs[0]);
    double *polygon_x_1;
    double *polygon_y_1;
    double *polygon_x_2;
    double *polygon_y_2;
    polygon_x_1 = mxGetPr(prhs[0]);
    polygon_y_1 = mxGetPr(prhs[1]);
    polygon_x_2 = mxGetPr(prhs[2]);
    polygon_y_2 = mxGetPr(prhs[3]);
    int M,N;
    M=mxGetM(prhs[0]);N=mxGetN(prhs[0]);//nV=M*N
	vector<Point> poly1;
	vector<Point> poly2;
	vector<Point> interPoly;
	
	for(int ii=0;ii<M*N;ii++)
	{
		Point tmp;
		tmp.x=polygon_x_1[ii];
		tmp.y=polygon_y_1[ii];
		poly1.push_back(tmp);
		//
		tmp.x=polygon_x_2[ii];
		tmp.y=polygon_y_2[ii];
		poly2.push_back(tmp);
		
	}
	//get convex hull
	PolygonClip(poly1,poly2,interPoly);
	//
	double area_int=0.0f;
	int nVertex=0;
	nVertex=interPoly.size();
	if (nVertex<= 0 ){ area_int=0.0f;}
	else{
		for(int ii=0;ii<nVertex;ii++){
			area_int=area_int + 0.5*(interPoly[ii].x*interPoly[(ii+1)%nVertex].y - interPoly[ii].y*interPoly[(ii+1)%nVertex].x);
		}
		area_int = fabs(area_int);
		
	}
	//area
	*int_area=area_int;

}





