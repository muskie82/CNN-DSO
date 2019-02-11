/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"
#include <chrono>


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
		int hl = hh>>lvl;
		points[lvl] = 0;
		numPoints[lvl] = 0;
		idepth[lvl] = new float[wl*hl];
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
		delete[] idepth[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}



void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian, cv::Mat depth)
{

    makeK(HCalib);
    firstFrame = newFrameHessian;

    PixelSelector sel(w[0],h[0]);

    float* statusMap = new float[w[0]*h[0]];
    bool* statusMapB = new bool[w[0]*h[0]];

    Mat33f K = Mat33f::Identity();
    K(0,0) = HCalib->fxl();
    K(1,1) = HCalib->fyl();
    K(0,2) = HCalib->cxl();
    K(1,2) = HCalib->cyl();

    float densities[] = {0.03,0.05,0.15,0.5,1};
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);


	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
    {
        sel.currentPotential = 3;
        int npts,npts_right;
        if(lvl == 0)
        {
            npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);

        }
        else
        {
            npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);
        }

        if(points[lvl] != 0) delete[] points[lvl];
        points[lvl] = new Pnt[npts];

        // set idepth map by static stereo matching. if no idepth is available, set 0.01.
        int wl = w[lvl], hl = h[lvl];
        Pnt* pl = points[lvl];
        int nl = 0;

    	float* depthmap_ptr = (float*)depth.data;

        for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
            for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
            {
                if(lvl==0 && statusMap[x+y*wl] != 0) {

                        pl[nl].u = x;
                        pl[nl].v = y;
                        pl[nl].idepth = 1.0f/(*(depthmap_ptr+(x+y*wl)));
                        pl[nl].iR = 1.0f/(*(depthmap_ptr+(x+y*wl)));

                        pl[nl].isGood=true;
                        pl[nl].energy.setZero();
                        pl[nl].lastHessian=0;
                        pl[nl].lastHessian_new=0;
                        pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
                        idepth[0][x+wl*y] = 1.0f/(*(depthmap_ptr+(x+y*wl)));

                        Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
                        float sumGrad2=0;
                        for(int idx=0;idx<patternNum;idx++)
                        {
                            int dx = patternP[idx][0];
                            int dy = patternP[idx][1];
                            float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
                            sumGrad2 += absgrad;
                        }

                        pl[nl].outlierTH = patternNum*setting_outlierTH;
                        nl++;
                        assert(nl <= npts);

                }

                if(lvl!=0 && statusMapB[x+y*wl])
                {
                    int lvlm1 = lvl-1;
                    int wlm1 = w[lvlm1];
                    float* idepth_l = idepth[lvl];
                    float* idepth_lm = idepth[lvlm1];
                    //assert(patternNum==9);
                    pl[nl].u = x+0.1;
                    pl[nl].v = y+0.1;
                    pl[nl].idepth = 0.1;
                    pl[nl].iR = 0.1;
                    pl[nl].isGood=true;
                    pl[nl].energy.setZero();
                    pl[nl].lastHessian=0;
                    pl[nl].lastHessian_new=0;
                    pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
                    int bidx = 2*x   + 2*y*wlm1;
                    idepth_l[x + y*wl] = idepth_lm[bidx] +
                                         idepth_lm[bidx+1] +
                                         idepth_lm[bidx+wlm1] +
                                         idepth_lm[bidx+wlm1+1];
					pl[nl].idepth = idepth_l[x + y*wl];
					pl[nl].iR =idepth_l[x + y*wl];

					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
                    float sumGrad2=0;
                    for(int idx=0;idx<patternNum;idx++)
                    {
                        int dx = patternP[idx][0];
                        int dy = patternP[idx][1];
                        float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
                        sumGrad2 += absgrad;
                    }

                    pl[nl].outlierTH = patternNum*setting_outlierTH;

                    nl++;
                    assert(nl <= npts);
                }


            }

        numPoints[lvl]=nl;
    }

    delete[] statusMap;
    delete[] statusMapB;

    makeNN();

    thisToNext=SE3();
    snapped = false;
    frameID = snappedAt = 0;

    for(int i=0;i<pyrLevelsUsed;i++)
        dGrads[i].setZero();

}


void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

